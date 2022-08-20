import torch
import torch.nn.functional as F
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
## test
import time

__author__ = 'yxy'

def compute_eer(trials, scores):
    '''
    @trials: a 2-dimensional list, including [[mod_id, test_id, 'target'/'nontarget'], [mod_id, test_id, 'target'/'nontarget'],......]
    @scores: a 2-dimensional list, including [[mod_id, test_id, socre], [mod_id, test_id, score],......] 
    trials order is equal to scores order, for example, trials[index][0] = scores[index][0], trials[index][1] = scores[index][1]
    @out: a 2-dimensional list, including [eer, threshold]
    '''
    assert len(trials) == len(scores)
    
    trials_len = len(trials)
    target_lst = []
    nontarget_lst = []
    for index in range(trials_len):
        if trials[index][2] == 'target':
            target_lst.append(scores[index][2])
        elif trials[index][2] == 'nontarget':
            nontarget_lst.append(scores[index][2])
        else:
            raise Exception("trials the last col value must be 'target' or 'nontarget'!!!!")
    
    target_len = len(target_lst)
    target_lst = sorted(target_lst)
    nontarget_len = len(nontarget_lst)
    nontarget_lst = sorted(nontarget_lst, reverse=True)

    for index in range(target_len-1):
        threshold = target_lst[index]
        frr = index / target_len
        non_pos = int(nontarget_len * frr)
        if nontarget_lst[non_pos] < threshold:
            break 
    return frr, threshold

def cosine_score(trials, mod_embeddings, test_embeddings, scores, n_jobs=1):
    '''
    @[in]trials: a 2-dimensional list, including [[mod_id, test_id, 'target'/'nontarget'], [mod_id, test_id, 'target'/'nontarget'],......]
    @[in]mod_embeddings: a 2-dimensional list, including [[mod_id, val1, val2, val3......], [mod_id, val1, val2, val3......],......]
    @[in]test_embeddings: a 2-dimensional list, including [[test_id, val1, val2, val3......], [test_id, val1, val2, val3......],......]
    @[out]scores: a 2-dimensional list, including [[mod_id, test_id, socre], [mod_id, test_id, score],......] 
    '''
    mod_dict = {}
    test_dict = {}
    for embedding in mod_embeddings:
        #mod_emb = [ float(val) for val in embedding[1:] ]
        mod_dict[embedding[0]] = torch.tensor(embedding[1:], dtype=torch.float32)
    for embedding in test_embeddings:
        #test_emb = [ float(val) for val in embedding[1:] ]
        test_dict[embedding[0]] = torch.tensor(embedding[1:], dtype=torch.float32)
    
    def compute_score(trial, mod_dict, test_dict):
        mod_id = trial[0]
        test_id = trial[1]
        score = F.cosine_similarity(mod_dict[mod_id].unsqueeze(-1), test_dict[test_id].unsqueeze(-1), dim=0)
        return mod_id, test_id, score[0].item()

    if n_jobs > 1:
        with joblib.parallel_backend('threading', n_jobs=n_jobs):
            temp_scores = Parallel(verbose=0)(delayed(compute_score)(trial, mod_dict, test_dict) for trial in tqdm(trials, total=len(trials),desc='compute score'))
    else:
        temp_scores = [compute_score(trial, mod_dict, test_dict) for trial in tqdm(trials, total=len(trials), desc='compute score')]
    scores.extend(temp_scores)

def compute_ser(labels, preds):
    '''
    @labels: a 1-dimensional list, including ['xxxx', 'xxxx', 'xxxx',......] 
    @preds: a 1-dimensional list, including ['xxxx', 'xxxx', 'xxxx',......]
    labels and preds list order is same, for example, labels[0] and preds[0] is for the same uttrance.
    '''
    assert len(labels) == len(preds)

    valid = 0
    for index, label in enmuerate(labels):
        if preds[index] == label:
            valid += 1
    return valid / len(labels), valid, len(labels)

# test
if __name__ == '__main__':
    #enroll_emb_path = '/nfs/user/yangxingya/raid0/company_task/task-tool/ivector_xvector/kaldi_ivector_16k16bit/exp/enroll_ivec_a'
    #verify_emb_path = '/nfs/user/yangxingya/raid0/company_task/task-tool/ivector_xvector/kaldi_ivector_16k16bit/exp/verify_ivec_a'
    enroll_emb_path = '/nfs/home/yangxingya/raid0/company_task/task56_mobile_asr_vpr/sugar/enroll_xvector.ark'
    verify_emb_path = '/nfs/home/yangxingya/raid0/company_task/task56_mobile_asr_vpr/sugar/test_xvector.ark'
    trials_path = '/nfs/home/yangxingya/raid0/company_task/task56_mobile_asr_vpr/sugar/test/DSD-V013-DIGIT-SV-TEST-MS-300s-v1.0/trials_same_session'

    trials_lst = []
    enroll_emb_lst = []
    verify_emb_lst = []
    with open(trials_path) as f_t, open(enroll_emb_path) as f_e, open(verify_emb_path) as f_v:
        for line in f_t.readlines():
            trials_lst.append(line.strip().split())

        for line in f_e.readlines():
            mod_id, left = line.strip().split('[')
            mod_id = mod_id.strip()
            left = [float(val) for val in left.split(']')[0].split()]
            enroll_emb_lst.append([mod_id]+left)

        for line in f_v.readlines():
            test_id, left = line.strip().split('[')
            test_id = test_id.strip()
            left = [float(val) for val in left.split(']')[0].split()]
            verify_emb_lst.append([test_id]+left)

    scores = []
     
    print("cosine score begin......")
    start_time = time.time()
    cosine_score(trials_lst, enroll_emb_lst, verify_emb_lst, scores, n_jobs=14)
    end_time = time.time()
    c_time = end_time - start_time
    print(f"cosine score end, cost time: {c_time}s")
    '''
    # scores_path = "/nfs/user/yangxingya/raid0/company_task/task-tool/ivector_xvector/kaldi_ivector_16k16bit/test/exp/score/scores"
    scores_path = "/nfs/home/yangxingya/raid0/company_task/company_git/vpr6.0/src/customized-compilation/demo-sv/exp-ivec2/scores/scores"
    with open(scores_path) as ff:
            for line in ff.readlines():
                mod_id, utt_id, score = line.strip().split()
                scores.append([mod_id, utt_id, float(score)])
    '''         
    print("compute eer begin......")
    start_time = time.time()
    eer,threshold = compute_eer(trials_lst, scores)
    end_time = time.time()
    c_time = end_time - start_time
    print(f"compute eer end, cost time: {c_time}s")

    print(f"eer: {eer}, threshold: {threshold}")

