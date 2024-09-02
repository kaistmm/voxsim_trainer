#!/bin/bash

python trainSpeakerNet.py \
    --model ecapa_tdnn \
    --nDataLoaderThread 4 \
    --max_epoch 30 \
    --batch_size 32 \
    --max_frames 300 \
    --eval_frames 400 \
    --scheduler steplr \
    --lr 1e-4 \
    --lr_decay 0.95 \
    --emb_dim 256 \
    --save_path exps_final/ecapa_mean_3 \
    --train_list data/voxsim_new/voxsim_train_list_mean.txt \
    --test_list data/voxsim_new/voxsim_test_list.txt \
    --train_path data/voxceleb1/ \
    --test_path data/voxceleb1/ \
    --seed 7 \
    --update_extract True \
    --initial_model ckpt/ecapa_tdnn.model
    # --distributed \
    # --port 7777 \