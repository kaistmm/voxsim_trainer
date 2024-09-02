#!/bin/bash

python trainSpeakerNet.py \
    --model wavlm_large \
    --nDataLoaderThread 4 \
    --max_epoch 10 \
    --batch_size 32 \
    --max_frames 300 \
    --eval_frames 400 \
    --scheduler steplr \
    --lr 1e-6 \
    --lr_decay 0.95 \
    --emb_dim 256 \
    --save_path exps_vcc/wavlm_ft_3 \
    --train_list data/vcc2018/vcc2018_sim_train_list.txt \
    --test_list data/vcc2018/vcc2018_sim_test_list.txt \
    --train_path data/vcc2018/wavs/ \
    --test_path data/vcc2018/wavs/ \
    --update_extract True \
    --max_label 4 \
    --seed 7 \
    --initial_model checkpoints/wavlm_ecapa.model \
    # --distributed \
    # --port 6060 \ 