import os

__author__ = 'yxy'

def split_text(text):
    ''' split "ab c" to [a,b,c] '''
    ret = []
    for i in text:
        if i != ' ':
            ret.append(i)
    return ret


def for_device_free(function):
    '''
    A decorator to make class-function with input-tensor device-free
    Used in TopVirtualNnet
    '''
    def wrapper(self, *tensor_sets):
        transformed = []
        for tensor in get_tensor(tensor_sets):
            transformed.append(to_device(self, tensor))
        return function(self, *transformed)
    return wrapper

def compute_ser(pred_dir, data_path, preds):
    '''
    predict: 
    for pred in preds:
        pred_path = os.path.join(pred_dir, pred)
    '''
    texts = []
    with open(data_path) as ff:
        line = ff.readline()
        while line:
            text = line.strip().split(',')[1]
            texts.append(text)  
            line = ff.readline()
    for pred in preds:
        pred_path = os.path.join(pred_dir, pred)
        with open(pred_path) as ff:
            valid = 0
            all_num = 0
            index = 0
            line = ff.readline()
            while line:
                if line.strip() == texts[index]:
                    valid += 1
                if line.strip() != "none":
                    all_num += 1
                index += 1
                line = ff.readline()    
    print(f"pred file: {pred}, all num: {all_num}, correct: {valid}, ser: {(all_num-valid)/all_num * 100}%")

