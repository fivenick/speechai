import torch
import traceback
from torch.utils.data import Dataset
import torch.nn as nn

#import sys
#sys.path.append("/nfs/home/yangxingya/raid0/company_task/self_project/speechai")

from libs.utils.frames import gen_window

__author__ = 'yxy'

class AsrDataset(Dataset):
    ''' self define asr dataset '''
    def __init__(self, file_path=None, vocab_path=None, fbank_dim=None, window_size=None, down_sample=None):
        super().__init__()
        self.vocab = {}
        self.file_path = file_path
        self.fbank_dim = fbank_dim
        self.window_size = window_size
        self.down_sample = down_sample
        self.info_lines = None
        self.line_num = -1 
        if vocab_path is not None:
            with open(vocab_path) as ff:
                for line in ff.readlines():
                    label,idx = line.strip().split(',') 
                    self.vocab[label] = int(idx)

    def __len__(self): 
        if self.line_num == -1: 
            info_lines = torch.load(self.file_path)
            self.line_num = len(info_lines) 
            del info_lines
        return self.line_num 
    
    def __getitem__(self, index): 
        '''
        the serialize data earch line format is: [idx, text, fbanks], text and fbanks is list too.
        '''
        if self.info_lines is None:
            self.info_lines = torch.load(self.file_path) 
        line = self.info_lines[index]
        feat, text = self.get_feat(line)
        
        if feat is not None and text is not None:
            return torch.tensor(feat, dtype=torch.double), torch.tensor(text), line[0]
        else:
            return None, None, line[0]
        
    def get_feat(self, line): 
        fbank_dim, window_size, down_sample = self.fbank_dim, self.window_size, self.down_sample
        labels = [self.vocab['<s>']] 
        ts = line[1]
        if len(ts) > 8:
            return None, None
        for label in ts:
            if not label in self.vocab: 
                return None, None
            labels.append(self.vocab[label])
        for i in range(8-len(ts)):
            labels.append(self.vocab['<blank>'])      
        labels.append(self.vocab['</s>'])
        
        frames_vals = line[2]
        #print(f"frames_vals shape: {torch.tensor(frames_vals).shape}")
        fbs = gen_window(frames_vals, fbank_dim, window_size, down_sample)
        
        #print(f"fbs shape: {torch.tensor(fbs).shape}")
        return fbs, labels
    # test
    def get(self):
        self.info_lines = torch.load(self.file_path)
        return self.info_lines

def collate_fn(data):
    ''' notice: some text length may be smaller than 8 (except <s> </s>) '''
    max_rows = 0
    for dd in data:
        if dd[0] is not None:
            max_rows = max(max_rows, (dd[0].shape)[0])
    text_res = None
    fbank_res = None
    idxs = []
    for dd in data:
        if dd[0] is None:
            continue
        left = max_rows - (dd[0].shape)[0]
        pad = nn.ZeroPad2d((0,0,0,left))
        out = pad(dd[0]).unsqueeze(0)
        idxs.append(dd[2])

        if fbank_res is None:
            fbank_res = out
            text_res = dd[1].unsqueeze(0)
        else:
            fbank_res = torch.cat([fbank_res, out], 0)
            text_res = torch.cat([text_res, dd[1].unsqueeze(0)], 0)

    return fbank_res, text_res, idxs

# test
'''
if __name__ == '__main__':
    dd = AsrDataset(file_path = "/nfs/home/yangxingya/raid0/company_task/self_project/speechai/data/serialize_data/asr_test_std/data_1")
    lines = dd.get()

    print(f"all line: {len(lines)}")
    print(f"index 0 idx: {lines[0][0]}")
    print(f"index 0 text: {lines[0][1]}")
    print(f"index 0 fbanks: {len(lines[0][2])}")
'''
