#!/bin/bash
uname -a
#date
#env
date

DATA_LIST_NAME='val.lst'
DATA_DIR='/media/szu/mydata/wuyong/datasets/cityscapes'
RESTORE_FROM='/media/szu/mydata/wuyong/snapshots/ccp_5/iter_20000.pth'
BATCH_SIZE=4
GPU='0,1,2,3'
OUTPUT_DIR='./ss_val_outputs'
INPUT_SIZE='769,769'


python ss_val_test_multigpu.py \
--data-dir ${DATA_DIR} \
--restore-from ${RESTORE_FROM} \
--data-list-name ${DATA_LIST_NAME} \
--batch-size ${BATCH_SIZE} \
--gpu ${GPU} \
--output-dir ${OUTPUT_DIR} \
--input-size ${INPUT_SIZE}


