#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# Command: bash run_scripts/muge_finetune_vit-b-16_rbt-base.sh ${DATAPATH}

# Number of GPUs per GPU worker
GPUS_PER_NODE=1
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=XX.XX.XX.XX
# The port for communication
export MASTER_PORT=8514
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0 

export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip/

DATAPATH=${1}

# data options
train_data=/home/ubuntu/cloudfs/ghost_data/newrank_hotrank_download/album_download/chinese_clip_training_til0123/chinese_clip_training_til0123_test_sample_1678055262.csv.tgz
val_data=/home/ubuntu/cloudfs/ghost_data/newrank_hotrank_download/album_download/chinese_clip_training_til0123/chinese_clip_training_til0123_test_sample_1678055262.csv.tgz # if val_data is not specified, the validation will be automatically disabled

# restore options
resume=/home/ubuntu/cloudfs/saved_models/Chinese_CLIP/ViT-H14/clip_cn_vit-h-14.pt # or specify your customed ckpt path to resume

reset_data_offset="--reset-data-offset"
reset_optimizer="--reset-optimizer"
# reset_optimizer=""

# output options
output_base_dir=/home/ubuntu/cloudfs/saved_models/Chinese_CLIP/experiments/
name=muge_finetune_vit-h-14_roberta-base_bs128_1gpu
save_step_frequency=999999 # disable it
save_epoch_frequency=1
log_interval=1
report_training_batch_acc="--report-training-batch-acc"
# report_training_batch_acc=""

# training hyper-params
context_length=52
warmup=100
batch_size=128
valid_batch_size=128
lr=5e-5
wd=0.001
max_epochs=3 # or you can alternatively specify --max-steps
valid_step_interval=150
valid_epoch_interval=1
vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese
use_augment="--use-augment"
# use_augment=""

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} cn_clip/training/main.py \
          --train-data=${train_data} \
          --val-data=${val_data} \
          --resume=${resume} \
          ${reset_data_offset} \
          ${reset_optimizer} \
          --logs=${output_base_dir} \
          --name=${name} \
          --save-step-frequency=${save_step_frequency} \
          --save-epoch-frequency=${save_epoch_frequency} \
          --log-interval=${log_interval} \
          ${report_training_batch_acc} \
          --context-length=${context_length} \
          --warmup=${warmup} \
          --batch-size=${batch_size} \
          --valid-batch-size=${valid_batch_size} \
          --valid-step-interval=${valid_step_interval} \
          --valid-epoch-interval=${valid_epoch_interval} \
          --lr=${lr} \
          --wd=${wd} \
          --max-epochs=${max_epochs} \
          --vision-model=${vision_model} \
          ${use_augment} \
          --text-model=${text_model}
