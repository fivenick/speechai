
cd /nfs/home/yangxingya/raid0/company_task/self_project/speechai 

NCCL_SOCKET_IFNAME=enp NCCL_DEBUG_SUBSYS=ALL \
python -m torch.distributed.launch --nproc_per_node=2 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="192.168.10.204" \
         --master_port=1234 \
         launcher/asr_train.py -train_file_num 30 -vocab ./data/asr_data_std/vocab.csv -train_data ./data/serialize_data/asr_train_std/data  -output_dir ./launcher/exp -log_dir ./launcher/exp/log -b 20 -warmup 8000 -log_name log_204.txt 
