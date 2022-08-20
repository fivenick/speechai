import os
import sys
import shutil
import argparse
import torch

sys.path.append('../libs/utils')
from utt import calc_fbank
from utt import run_cmd

__author__ = 'yxy'

vad_data_dir = './vad/.data'
vad_res = 'vad.res'

def get_frame_num(time_dur, params): 
    frames_num = int((time_dur * 1000 - params["frame_length"]) / params["frame_shift"]) + 1
    return frames_num

def check_duration(fbanks, opt, params):
    ''' the dur time must larger than min_time, less than max_time  '''
    frames_num = len(fbanks)
    if opt.min_time is not None:
        if frames_num <= get_frame_num(opt.min_time, params): 
            return False
    if opt.max_time is not None:
        if frames_num >= get_frame_num(opt.max_time, params):
            return False
    return True

def init_frames_valid(frames_valid, vad_path):
    with open(vad_path) as f:
        line = f.readline()
        while line:
            utt_id, vads = line.strip().split('[')
            utt_id = utt_id.strip()
            vads = vads.split(']')[0].strip().split()
            frames_valid[utt_id] = vads
            line = f.readline()

def vad_fbank(fbanks, vads, params):
    ''' vad the fbank '''
    assert len(fbanks) == len(vads)
    ret = []
    for i in range(len(vads)):
        if vads[i] == '1':
            ret.append(fbanks[i])
    return ret    
 
def save_fbank(opt):
    ''' fbank for 16k16bit wav, save the wav fbanks '''
    params = {
        'num_mel_bins':80,
        'frame_length':25,
        'frame_shift':10
    }
    with open(opt.data_scp_path) as fd, open(opt.text_scp_path) as ft:
        global vad_data_dir
        lines_num = 0
        utt_dict = {}
        line = fd.readline()
        while line:
            lines_num += 1
            idx, utt_path = line.strip().split()
            assert not idx in utt_dict
            utt_dict[idx] = utt_path
            line = fd.readline()
            
        assert opt.save_files_num != 0

        lines_per_file = int(lines_num / opt.save_files_num)

        frames_valid = {}
        if opt.vad:
            init_frames_valid(frames_valid, os.path.join(vad_data_dir, vad_res))
            shutil.rmtree(vad_data_dir)
        
        file_index = 1
        line_index = 0
        save_lst = []
        line = ft.readline()
        while line:
            line_index += 1
            idx, *text = line.strip().split()
            
            if not idx in utt_dict:
                line = ft.readline()
                continue

            utt_path = utt_dict[idx]
            fbanks = calc_fbank(utt_path, params)
            
            if fbanks is not None:
                if opt.vad:
                    fbanks = vad_fbank(fbanks, frames_valid[idx], params) 
                    if check_duration(fbanks, opt, params):
                        save_lst.append([idx, text, fbanks])

            if (line_index % lines_per_file) == 0 and file_index < opt.save_files_num:
                file_path = opt.save_path
                if opt.save_files_num > 1:
                    file_path = f"{file_path}_{file_index}"
                torch.save(save_lst, file_path)
                save_lst = []
                file_index += 1
                
            line = ft.readline()

        file_path = opt.save_path
        if opt.save_files_num > 1:
            file_path = f"{file_path}_{file_index}"
        torch.save(save_lst, file_path)

def vad(opt):
    ''' the wav should be 16k16bit '''
    global vad_data_dir
    if os.path.exists(vad_data_dir):
        shutil.rmtree(vad_data_dir)
    os.makedirs(vad_data_dir)

    des_path = os.path.join(vad_data_dir, 'wav.scp')
    shutil.copy(opt.data_scp_path, des_path)
    vad_data_dir = os.path.abspath(vad_data_dir)
    print("mfcc and vad... ...")
    cmd_str = f'sh vad/mfcc_vad.sh {vad_data_dir} {vad_res}' 
    out_info, err_info = run_cmd(cmd_str)
    print(err_info)

def main_deal_data():
    '''
    usage: python asr_data_preprocess.py -data_scp_path train_data.scp -text_scp_path train_text.csv -save_path ./data/train_data.pkl -save_files_num 3 -vad True
    the utt is 16k16bit
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_scp_path', required=True, type=str)
    parser.add_argument('-text_scp_path', required=True, type=str)
    parser.add_argument('-save_path', required=True, type=str)
    parser.add_argument('-save_files_num', type=int, default=1)
    parser.add_argument('-min_time', type=float, default=None)
    parser.add_argument('-max_time', type=float, default=None)
    parser.add_argument('-vad', action='store_true', default=False)

    opt = parser.parse_args()

    print(opt)

    save_dir = os.path.dirname(opt.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if opt.vad:
        vad(opt)
    
    save_fbank(opt)

if __name__ == '__main__':
    main_deal_data()
