
cd /nfs/home/yangxingya/raid0/company_task/task56_mobile_asr_vpr/extremely-low-footprint-end-to-end-asr-system-for-smart-device 

NCCL_SOCKET_IFNAME=enp NCCL_DEBUG_SUBSYS=ALL \
python -m torch.distributed.launch --nproc_per_node=1 \
         --nnodes=2 \
         --node_rank=0 \
         --master_addr="192.168.10.209" \
         --master_port=1234 \
         train.py -train_file_num 100 -vocab ./data/vocab.csv -train_data .temp/serialize_train_data/train_text.csv -val_data .temp/val_text.csv -test_data .temp/test_text.csv_sub100 -output_dir ./exp -log_dir ./log -b 100 -warmup 128000 -log_name log_209.txt 
