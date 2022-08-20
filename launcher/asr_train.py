import os
import sys
import time
import math
import logging
import argparse
import dill as pickle
import torch
#from torchtext.legacy.data import Field, Dataset, BucketIterator
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS 

from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import libs.training.asr_dataset as ad 
from model.asr_models import Transformer
from libs.training.optim import ScheduledOptim
from libs.utils.frames import gen_window
from libs.training.asr_dataset import collate_fn

__author__ = 'yxy'



def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    #trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def train_epoch(model, train_dls, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 

    desc = '  - (Training)   '

    accumulation_steps = 15

    logging.info('start loading data to train_dls')
    train_dls = load_spec_data(opt.train_data, opt, device, files_num=opt.train_file_num)
    logging.info('end loading data to train_dls')
    train_dls_len = len(train_dls)

    print(f"train dl len: {train_dls_len}")
    #for train_dl in train_dls:
    for ii in range(train_dls_len):

        train_dl = train_dls[0] 

        index = 1 

        for batch in tqdm(train_dl, mininterval=2, desc=desc, leave=False):

            if batch[0] is None or batch[1] is None:
                continue

            logging.info(f"in train data shape: {batch[0].shape}")
            logging.info(f"in train text shape: {batch[1].shape}")
            # prepare data
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch[1], opt.trg_pad_idx))

            # forward
            pred = model(batch[0].to(device), trg_seq)

            # backward and update parameters
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=smoothing) 

            loss = loss / accumulation_steps 

            loss.backward()

            if index % accumulation_steps == 0:
                optimizer.step_and_update_lr()
                optimizer.zero_grad()

            index += 1

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

        optimizer.step_and_update_lr()
        optimizer.zero_grad()
        del train_dls[0]

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, val_dl, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(val_dl, mininterval=2, desc=desc, leave=False):

            # prepare data
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch[1], opt.trg_pad_idx))

            # forward
            pred = model(batch[0].to(device), trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, train_dls, val_dl, optimizer, device, opt):
    ''' start training '''
    global tboard
    if opt.use_tb:
        logging.info("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))
    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    logging.info('[Info] Training performance will be written to file: {} and {}'.format(log_train_file, log_valid_file))
    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')
    def print_performances(header, ppl, accu, start_time, lr):
        logging.info('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=ppl,
                  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))
    valid_losses = []
    for epoch_i in range(opt.epoch):
        logging.info(f'[ Epoch {epoch_i} ]')


        start = time.time()
        train_loss, train_accu = train_epoch(
            model, train_dls, optimizer, opt, device, smoothing=opt.label_smoothing)


        train_ppl = math.exp(min(train_loss, 100))
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, start, lr)

        start = time.time()
        #valid_loss, valid_accu = eval_epoch(model, val_dl, device, opt)
        valid_loss = 1.4
        valid_accu = 0.23

        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.rank == '0':
            if opt.save_mode == 'all':
                #model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                model_name = f'model_epoch_{epoch_i}.chkpt'
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = 'model.chkpt'
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                    logging.info('    - [Info] The checkpoint file has been updated.')
        '''
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100*train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100*valid_accu))

        if opt.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)
        '''

def load_spec_data(data_path, opt, device, files_num=None):
    if files_num is not None:
        dl_lst = []
        for i in range(1, files_num+1):
            new_path = f"{data_path}_{i}" 
            new_path = os.path.abspath(new_path)
            ds = ad.AsrDataset(file_path=new_path, vocab_path=opt.vocab, fbank_dim=opt.fbank_dim, window_size=opt.window_size, down_sample=opt.down_sample)
            data_sampler = DS(ds)
            dl = DataLoader(ds, collate_fn=collate_fn, batch_size=opt.batch_size, num_workers=2, sampler=data_sampler)
            dl_lst.append(dl) 
        return dl_lst
    else:
        ds = ad.AsrDataset(file_path=data_path, vocab_path=opt.vocab, fbank_dim=opt.fbank_dim, window_size=opt.window_size, down_sample=opt.down_sample)
        data_sampler = DS(ds)
        dl = DataLoader(ds, collate_fn=collate_fn, batch_size=opt.batch_size, num_workers=2, sampler=data_sampler)
        return dl    

def load_data(opt, device):
    '''
    if not "train_data" in opt:
        train_dls = None
    else:
        train_dls = load_spec_data(opt.train_data, opt, device, files_num=opt.train_file_num)
    '''
    train_dls = None
    '''
    if not "val_data" in opt:
        val_dl = None
    else:
        val_dl = load_spec_data(opt.val_data, opt, device)
    '''
    val_dl = None
    if not "test_data" in opt:
        test_dl = None
    else:
        test_dl = load_spec_data(opt.test_data, opt, device)
    return train_dls, val_dl, test_dl

def load_text_data(opt, device):
    data = pickle.load(open(opt.data_pkl, 'rb')) 
    fields = {'id':data['id'], 'text':data['text'], 'fbank':data['fbank']}
    train_ds = Dataset(examples=data['train'], fields=fields)
    val_ds = Dataset(examples=data['val'], fields=fields)
    test_ds = Dataset(examples=data['test'], fields=fields)
    train_iter = BucketIterator(train_ds, batch_size = 1, device=device, train=True)
    val_iter = BucketIterator(val_ds, batch_size = 1, device=device)
    test_iter = BucketIterator(test_ds, batch_size = 1, device=device)
    return train_iter, val_iter, test_iter

def train_main():
    '''
    usage:
    python train.py -vocab ./vocab.csv -train_data ./data/train_data.csv -val_data ./data/val_data.csv -test_data ./data/test_data.csv -output_dir ./ -log_dir ./ -b 256 -warmup 128000
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-log_dir', type=str, default=None)
    parser.add_argument('-log_name', type=str, default=None)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=8000)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-seed', type=int, default=10)
    parser.add_argument('-fbank_dim', type=int, default=80)
    parser.add_argument('-fbank_frame_dim', type=int, default=320)
    parser.add_argument('-down_sample', type=int, default=80)
    parser.add_argument('-train_data', required=True, type=str)
    parser.add_argument('-val_data', type=str, default=None)
    parser.add_argument('-test_data', type=str, default=None)
    parser.add_argument('-vocab', required=True, type=str)
    parser.add_argument('-window_size', type=int, default=4)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')
    parser.add_argument('-d_model', type=int, default=320)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head',type=int, default=3)
    parser.add_argument('-n_layers', type=int, default=3)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-epoch', type=int, default=12)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='all')
    parser.add_argument('-train_file_num', type=int, default=26)
    parser.add_argument('-n_trg_vocab', type=int, default=15)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('-n_enc_layers', type=int, default=1)
    parser.add_argument('-n_enc_sublayers', type=int, default=8)
    parser.add_argument('-n_dec_layers', type=int, default=1)
    parser.add_argument('-n_dec_sublayers', type=int, default=3)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    opt.trg_pad_idx = 11 

    if opt.log_dir is not None:
        logging.basicConfig(filename = os.path.join(opt.log_dir, opt.log_name), level=logging.INFO)
    
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        #torch.backends_cudnn.benchmark = False
        np.random.seed(opt.seed)
        random.seed(opt.seed)
    if opt.output_dir is None:
        logging.info('No experiment result will be saved.')
        raise
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    dist.init_process_group(backend='nccl')   
    #local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = opt.local_rank
    opt.rank = os.environ['RANK']

    device = f"cuda:{local_rank}"
    logging.info(f"local_rank: {local_rank} start training ........")
    #rank = int(os.environ["RANK"])
    #print(f"rank: {rank}, local_rank: {local_rank} start training ........")

    #device = torch.device('cuda' if opt.cuda else 'cpu') 
    logging.info(f"device: {device}")
    #train_dls, val_dl, test_dl = load_data(opt, device)
    train_dls=None
    val_dl=None
    test_dl=None

    transformer = Transformer(
        n_trg_vocab=opt.n_trg_vocab,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj,
        n_enc_layers=opt.n_enc_layers,
        n_enc_sublayers=opt.n_enc_sublayers,
        n_dec_layers=opt.n_dec_layers,
        n_dec_sublayers=opt.n_dec_sublayers).to(device)
    transformer = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    optimizer = ScheduledOptim(optim.Adam(transformer.parameters(), betas=(0.9, 0.999), eps=1e-09), opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(transformer, train_dls, val_dl, optimizer, device, opt)
    logging.info('end train........')
   
## global variables
tboard = SummaryWriter()
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    train_main()
