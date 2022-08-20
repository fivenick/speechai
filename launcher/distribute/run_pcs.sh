

ip_lst=('209' '204')
suffix_idx=('0' '1')

ip_len=${#ip_lst[@]}

root_dir=/nfs/home/yangxingya/raid0/company_task/task56_mobile_asr_vpr/extremely-low-footprint-end-to-end-asr-system-for-smart-device
run_dir=${root_dir}/distribute
log_dir=${root_dir}/log

for i in $(seq 1 ${ip_len})
do
  index=$(($i-1))
  sshpass -p Jackand7315... ssh yangxingya@192.168.10.${ip_lst[$index]} "nohup sh ${run_dir}/run_node${suffix_idx[$index]}.sh > ${log_dir}/sshpass_${ip_lst[$index]}.txt 2>&1" &
done
