#export KALDI_ROOT=/nfs/user/shared/kaldi-base/beta/with-cuda
#export KALDI_ROOT=/nfs/user/shared/kaldi-base/191121/with-cuda
#export KALDI_ROOT=/nfs/share/kaldi-astar
export KALDI_ROOT=/nfs/data/disk05/kaldi-base/current/with-cuda
export LD_LIBRARY_PATH=${KALDI_ROOT}/src/lib
export LD_LIBRARY_PATH=${KALDI_ROOT}/tools/openfst-1.6.7/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${KALDI_ROOT}/tools/OpenBLAS/install/lib:${LD_LIBRARY_PATH}
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
