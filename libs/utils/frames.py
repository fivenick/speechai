import numpy as np
__author__ = 'yxy'

def down_sample(ds, frame):
    del_idxs = np.random.choice(len(frame), ds, replace=False)
    del_idxs.sort()
    for index,item in enumerate(del_idxs):
        frame.pop(item-index)

def gen_window(frames, feat_dim, window_size, ds):
    ''' 
    frames shape [frames, dim]: [[1,2],[3,......]] 
    window_size=3,feat_dim=2
    convert frames [[1,2],[3,4],[5,6],[7,8],[9,10]] to [[1,2,3,4,5,6],[3,4,5,6,7,8],[5,6,7,8,9,10]] 
    than downsample=2, means delete 2 items for one row
    [[1,2,4,5],[4,5,6,7],[5,7,8,9]]
    '''
    ret = []
    flen = len(frames)
    start = 0
    end = window_size
    while end < flen:
        temp = []
        for index in range(start, end):
            temp.extend(frames[index]) 
        ret.append(temp)
        start += 1
        end += 1
    return ret
    
