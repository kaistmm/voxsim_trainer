#!/bin/bash

python trainSpeakerNet.py \
    --model wavlm_large \
    --nDataLoaderThread 4 \
    --max_epoch 10 \
    --batch_size 32 \
    --max_frames 300 \
    --eval_frames 400 \
    --scheduler steplr \
    --lr 1e-3 \
    --lr_decay 0.95 \
    --emb_dim 256 \
    --train_list data/voxsim_new/voxsim_train_list_mean.txt \
    --test_list data/voxsim_new/voxsim_test_list.txt \
    --train_path data/voxceleb1/ \
    --test_path data/voxceleb1/ \
    --save_path exps/pretrain \
    # --distributed \
    # --port 7777 \