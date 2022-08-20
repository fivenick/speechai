#!/usr/bin/env python

import os
import sys



def _main():
	wav_scp, spk2utt, utt2spk = sys.argv[1:] 
	with open(wav_scp) as ff_wav, open(spk2utt, 'w') as ff_spk2utt, open(utt2spk, 'w') as ff_utt2spk:
		spk2utt_dict = dict()
		for line in ff_wav.readlines():
			utt_id = line.split()[0]	
			spk_id = utt_id.split('_')[0]
			if not spk_id in spk2utt_dict:
				spk2utt_dict[spk_id] = [spk_id]
			spk2utt_dict[spk_id].append(utt_id)
			ff_utt2spk.write(utt_id + ' ' + spk_id + '\n')
		for key in spk2utt_dict:
			ff_spk2utt.write(' '.join(spk2utt_dict[key]) + '\n')

if __name__ == '__main__':
	_main()

