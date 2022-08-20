#!/bin/bash


cd ./vad
. ./cmd.sh
. ./path.sh


# data aug digit train data
log_root='./exp/log'
vaddir='./exp/vad'
mfccdir='./exp/mfcc'
data_dir=$1
vad_res=$2

sample_rate=16000

stage=0

if [ -d ${mfccdir} ]; then
  rm -rf ${mfccdir}
fi
if [ -d ${vaddir} ]; then
  rm -rf ${vaddir}
fi
if [ -d ${log_root} ]; then
  rm -rf ${log_root}
fi

mkdir -p ${mfccdir}
mkdir -p ${vaddir}
mkdir -p ${log_root}

if [ $stage -le 0 ]; then
  echo "gen spk2utt and utt2spk from wav.scp......"
  python wav_spk2utt_utt2spk.py ${data_dir}/wav.scp ${data_dir}/spk2utt ${data_dir}/utt2spk
fi

if [ $stage -le 1 ]; then
  echo "sort wav.scp utt2spk spk2utt......."
  for file in wav.scp utt2spk spk2utt
  do
    sort -k1 ${data_dir}/$file > temp
    mv temp ${data_dir}/$file
  done
fi

if [ $stage -le 2 ]; then
  echo "mfcc is going........"
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 1 --cmd "$train_cmd" \
    $data_dir ${log_root}/make_mfcc $mfccdir
  utils/fix_data_dir.sh $data_dir

  echo "vad is going........."
  sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
    $data_dir ${log_root}/make_vad $vaddir
fi

if [ $stage -le 3 ]; then
  echo "remove the former vad res......"
  if [ -f ${data_dir}/${vad_res} ]; then
    rm -f ${data_dir}/${vad_res}
  fi

  echo "gen new vad res......"
  arks=`find ${vaddir} -name "*.ark"`
  for t_ark in $arks 
  do
    copy-vector --binary=false ark:${t_ark} ark,t:- >> ${data_dir}/${vad_res}
  done
fi

if [ $stage -le 4 ]; then
  if [ -d ${mfccdir} ]; then
    rm -rf ${mfccdir} 
  fi
  if [ -d ${vaddir} ]; then
    rm -rf ${vaddir} 
  fi
fi
