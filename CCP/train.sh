#!/bin/bash
uname -a
#date
#env
date

BATCH_SIZE=8
DATA_DIR='/home/xieke/wuyong/data/datasets/cityscapes'
START_ITERS=0
NUM_SETPS=40000
RESTORE_FROM='../resnet101-imagenet.pth'
SNAPSHOT_DIR='/media/szu/mydata/wuyong/snapshots/ccp_3'
GPU='0,1,2,3'
WARMUP_STEPS=5000
WARMUP_START_LR=0.001
INPUT_SIZE='769,769'
TRAIN_LIST_NAME='train.lst'

python train.py \
--batch-size ${BATCH_SIZE} \
--data-dir ${DATA_DIR} \
--start-iters ${START_ITERS} \
--num-steps ${NUM_SETPS} \
--restore-from ${RESTORE_FROM} \
--snapshot-dir ${SNAPSHOT_DIR} \
--gpu ${GPU} \
--warmup-steps ${WARMUP_STEPS} \
--warmup-start-lr ${WARMUP_START_LR} \
--input-size ${INPUT_SIZE} \
--train-list-name ${TRAIN_LIST_NAME}
