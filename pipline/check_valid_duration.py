
import os
import sys
import argparse

sys.path.append('../libs/utils')
from utt import get_duration

__author__ = 'yxy'

def check_valid_duration(opt):
    with open(opt.data_scp_path) as fs, open(opt.save_path, 'w') as fw:
        line = fs.readline()
        while line:
            utt_id, utt_path = line.strip().split()

            # check the utt is valid or not
            if not os.path.exists(utt_path):
                line = fs.readline()
                continue

            # check the utt duration
            utt_dur = get_duration(utt_path)
            if opt.min_time is not None:
                if utt_dur < float(opt.min_time):
                    line = fs.readline()
                    continue
            if opt.max_time is not None:
                if utt_dur > float(opt.max_time):
                    line = fs.readline()
                    continue

            fw.write(line)

            line = fs.readline()

def run_main():
    '''
    usage: python check_valid_duration.py -data_scp_path ./train.scp -save_path ./new_train.scp -min_time 2.3 -max_time 8.0
    delete the invalid utt, and time duration less than min_time, or larger than max_time.
    min_time, max_time unit is seconds
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_scp_path', required=True, type=str)
    parser.add_argument('-save_path', required=True, type=str)
    parser.add_argument('-min_time', type=float, default=None)
    parser.add_argument('-max_time', type=float, default=None)

    opt = parser.parse_args()

    check_valid_duration(opt)

if __name__ == '__main__':
    run_main()
