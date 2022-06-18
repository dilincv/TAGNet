#!/bin/bash
uname -a
#date
#env
date

MODE='test'
DATA_DIR='/media/szu/mydata/wuyong/datasets/cityscapes'
DATA_LIST_ROOT='./dataset/list/cityscapes/'
RESTORE_FROM='/media/szu/mydata/wuyong/snapshots/baseline_pt0_4_mod_train_scale/CS_scenes_40000.pth'
GPU='0,1,2,3'
OUTPUT_DIR='./ms_test_ori_outputs'

python ms_val_test_ori.py \
--data-dir ${DATA_DIR} \
--restore-from ${RESTORE_FROM} \
--mode ${MODE} \
--data-list-root ${DATA_LIST_ROOT} \
--gpu ${GPU} \
--output-dir ${OUTPUT_DIR} \

