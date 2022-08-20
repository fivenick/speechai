''' Translate input text with trained model. '''
import os
import sys
import torch
import random
import numpy as np
import argparse
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed

import time
from torch.utils.data import DataLoader
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import libs.utils.asr_dict_constant as Constants
from model.asr_models import Transformer
from libs.decoder.asr_decoder import Translator
import libs.training.asr_dataset as ad
from libs.training.asr_dataset import collate_fn

def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']
    
    model_param = checkpoint['model']
    new_model_param = {}
    for k,v in model_param.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        new_model_param[new_k] = v

    model = Transformer(
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

    model.load_state_dict(new_model_param)
    #model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 

def load_spec_data(data_path, opt, device):
    dls = []
    for i in range(1, opt.test_file_num + 1):
        test_path = f"{data_path}_{i}" 
        ds = ad.AsrDataset(file_path=test_path, vocab_path=opt.vocab, fbank_dim=opt.fbank_dim, window_size=opt.window_size, down_sample=opt.down_sample)
        dl = DataLoader(ds, collate_fn=collate_fn, batch_size=opt.batch_size, num_workers=2)
        dls.append(dl)
    return dls

def init_vocab(opt, vocab, reverse_vocab):
    with open(opt.vocab) as ff:
        for line in ff.readlines():
            label, digit = line.strip().split(',')
            vocab[int(digit)] = label
            reverse_vocab[label] = int(digit)

def recognize(example, f, translator, valid, device, all_time, vocab):
    src_seq = example[0]
    if src_seq is None:
        f.write('none \n')
        return

    idx = example[2][0]

    start_time = time.time()
    
    pred_seq = translator.translate_sentence(src_seq.to(device))
    end_time = time.time()
    all_time.append(end_time - start_time)
    valid.append(1)

    pred_line = ' '.join(vocab[idx] for idx in pred_seq)
    pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
    #print(pred_line)
    f.write(idx + ' ' + pred_line.strip() + '\n')

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='asr.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=9)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-test_data', required=True, type=str)
    parser.add_argument('-vocab', required=True, type=str)
    parser.add_argument('-fbank_dim', type=int, default=80)
    parser.add_argument('-window_size', type=int, default=4)
    parser.add_argument('-down_sample', type=int, default=80)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-seed', type=int, default=10)
    parser.add_argument('-test_file_num', type=int, default=2)
    parser.add_argument('-n_enc_layers', type=int, default=1)
    parser.add_argument('-n_enc_sublayers', type=int, default=8)
    parser.add_argument('-n_dec_layers', type=int, default=1)
    parser.add_argument('-n_dec_sublayers', type=int, default=3)
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')
    parser.add_argument('-n_trg_vocab', type=int, default=15)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head',type=int, default=3)
    parser.add_argument('-d_model', type=int, default=320)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-n_jobs', type=int, default=1)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    ''' 
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
    ''' 
    test_dls = load_spec_data(opt.test_data, opt, device) 
    #print(f"test_dl shape: {next(iter(test_dl))[0].shape}")
    #print(f"test_dl shape: {next(iter(test_dl))[1].shape}")
    
    vocab = {}
    reverse_vocab = {}
    init_vocab(opt, vocab, reverse_vocab)

    opt.trg_pad_idx = reverse_vocab['<blank>']
    opt.trg_bos_idx = reverse_vocab['<s>']
    opt.trg_eos_idx = reverse_vocab['</s>']
    opt.src_pad_idx = 0

    translator = Translator(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).to(device)
    
    all_time = []
    valid = []
    with open(opt.output, 'w') as f:     
        with joblib.parallel_backend('threading', n_jobs=opt.n_jobs):
            for i in range(opt.test_file_num):
                test_dl = test_dls[0]
                Parallel(verbose=0)(delayed(recognize)(example, f, translator, valid, device, all_time, vocab) for example in tqdm(test_dl, total=len(test_dl), mininterval=2, desc='recognize'))
                del test_dls[0] 

    print('[Info] Finished.')
    utt_num = len(valid)
    all_time = sum(all_time)
    print(f"all time: {all_time}, utt nums: {utt_num}, avg: {all_time / utt_num}")

if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    main()
