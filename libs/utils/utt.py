import torchaudio
import subprocess

__author__ = 'yxy'

def run_cmd(cmd_str):
    res_info = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out_info, err_info = res_info.communicate()
    return out_info.decode(), err_info.decode()

def gen_seconds(info):
    items = info.split(':')
    seconds = 0.0
    for i in items:
        seconds = seconds * 60 + float(i)
    return seconds

def get_duration(utt_path):
    ''' 
    get the utt duration.
    @output: the utt_path duration, seconds
    '''
    cmd_str = f"soxi -d {utt_path}"
    out_info, _ = run_cmd(cmd_str)
    return gen_seconds(out_info)

def calc_fbank(utt_path, params):
    ''' 
    calcualte the wav fbank.
    fbanks dim is: [frames, frame dim]
    '''
    try:
        wavform, sample_rate = torchaudio.load(utt_path)
        fbanks = torchaudio.compliance.kaldi.fbank(wavform, **params)
        return fbanks.tolist()
    except Exception as e:
        print(f"error: fbank wav: {utt_path}")
        print(e.args)
        return None

