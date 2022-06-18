#!/bin/bash
uname -a
#date
#env
date

DATA_LIST_NAME='val.lst'
DATA_DIR='/media/szu/mydata/wuyong/datasets/cityscapes'
DATA_LIST_ROOT='./dataset/list/cityscapes/'
RESTORE_FROM='/media/szu/mydata/wuyong/snapshots/baseline_pt0_4_mod_train_scale/CS_scenes_40000.pth'
GPU='0,1,2,3'
INPUT_SIZE='769,769'
OUTPUT_DIR='./ms_val_yf_outputs'

python ms_val_test_yf.py \
--data-dir ${DATA_DIR} \
--restore-from ${RESTORE_FROM} \
--data-list-name ${DATA_LIST_NAME} \
--data-list-root ${DATA_LIST_ROOT} \
--gpu ${GPU} \
--input-size ${INPUT_SIZE} \
--output-dir ${OUTPUT_DIR}
