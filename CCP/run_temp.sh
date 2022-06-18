#!/bin/bash

TRAIN_BATCH_SIZE=8
DATA_DIR='/home/ubuntu/cityscapes'
START_ITERS=10000
NUM_SETPS=40000
TRAIN_RESTORE_FROM='/home/ubuntu/CCP/CCP51/snapshots/iter_10000.pth'
SNAPSHOT_DIR='./snapshots'
GPU='4,5,6,7'
WARMUP_STEPS=7000
WARMUP_START_LR=0.0001
INPUT_SIZE='769,769'

TRAIN_LIST_NAME='train.lst'
VAL_LIST_NAME='val.lst'
TEST_LIST_NAME='test.lst'
VAL_TEST_RESTORE_FROM='./snapshots/iter_47000.pth'
VAL_TEST_RESTORE_FROM2='./snapshots/iter_46750.pth'
VAL_TEST_RESTORE_FROM3='./snapshots/iter_46500.pth'
VAL_TEST_RESTORE_FROM4='./snapshots/iter_46250.pth'
VAL_TEST_RESTORE_FROM5='./snapshots/iter_46000.pth'
VAL_TEST_RESTORE_FROM7='./snapshots/iter_44000.pth'
VAL_TEST_RESTORE_FROM8='./snapshots/iter_43000.pth'
VAL_TEST_RESTORE_FROM9='./snapshots/iter_42000.pth'
VAL_TEST_BATCH_SIZE=4

VAL_OUTPUT_DIR_MS='./snapshots/val_output/ms_val_47000'
VAL_OUTPUT_DIR_MS2='./snapshots/val_output/ms_val_46750'
VAL_OUTPUT_DIR_MS3='./snapshots/val_output/ms_val_46500'
VAL_OUTPUT_DIR_MS4='./snapshots/val_output/ms_val_46250'
VAL_OUTPUT_DIR_MS5='./snapshots/val_output/ms_val_46000'
VAL_OUTPUT_DIR_MS7='./snapshots/val_output/ms_val_44000'
VAL_OUTPUT_DIR_MS8='./snapshots/val_output/ms_val_43000'
VAL_OUTPUT_DIR_MS9='./snapshots/val_output/ms_val_42000'

python ms_val_test_yf.py    \
--data-dir ${DATA_DIR}  \
--restore-from ${VAL_TEST_RESTORE_FROM} \
--data-list-name ${VAL_LIST_NAME}   \
--gpu ${GPU} \
--output-dir ${VAL_OUTPUT_DIR_MS}

python ms_val_test_yf.py    \
--data-dir ${DATA_DIR}  \
--restore-from ${VAL_TEST_RESTORE_FROM2} \
--data-list-name ${VAL_LIST_NAME}   \
--gpu ${GPU} \
--output-dir ${VAL_OUTPUT_DIR_MS2}

python ms_val_test_yf.py    \
--data-dir ${DATA_DIR}  \
--restore-from ${VAL_TEST_RESTORE_FROM3} \
--data-list-name ${VAL_LIST_NAME}   \
--gpu ${GPU} \
--output-dir ${VAL_OUTPUT_DIR_MS3}

python ms_val_test_yf.py    \
--data-dir ${DATA_DIR}  \
--restore-from ${VAL_TEST_RESTORE_FROM4} \
--data-list-name ${VAL_LIST_NAME}   \
--gpu ${GPU} \
--output-dir ${VAL_OUTPUT_DIR_MS4}

python ms_val_test_yf.py    \
--data-dir ${DATA_DIR}  \
--restore-from ${VAL_TEST_RESTORE_FROM5} \
--data-list-name ${VAL_LIST_NAME}   \
--gpu ${GPU} \
--output-dir ${VAL_OUTPUT_DIR_MS5}

python ms_val_test_yf.py    \
--data-dir ${DATA_DIR}  \
--restore-from ${VAL_TEST_RESTORE_FROM7} \
--data-list-name ${VAL_LIST_NAME}   \
--gpu ${GPU} \
--output-dir ${VAL_OUTPUT_DIR_MS7}

python ms_val_test_yf.py    \
--data-dir ${DATA_DIR}  \
--restore-from ${VAL_TEST_RESTORE_FROM8} \
--data-list-name ${VAL_LIST_NAME}   \
--gpu ${GPU} \
--output-dir ${VAL_OUTPUT_DIR_MS8}

python ms_val_test_yf.py    \
--data-dir ${DATA_DIR}  \
--restore-from ${VAL_TEST_RESTORE_FROM9} \
--data-list-name ${VAL_LIST_NAME}   \
--gpu ${GPU} \
--output-dir ${VAL_OUTPUT_DIR_MS9}