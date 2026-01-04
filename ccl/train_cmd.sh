#!/bin/bash

python run_ccl.py \
--exp_name $1 \
--wandb_project CCL_v2 \
--traindom $2 \
--testdoms $2 \
--n_bat 128 \
--n_epk 100 \
--lr 1e-4 \
--wl2 1e-2 \
--wgen 1e-6 \
--wlogpi 1 \
--wrecon_x 1 \
--wrecon_e 1 \
--wrecon_c 1e-2 \
--wrecon_s 1e-2 \
--sig_c 1. \
--sig_s 0.5 \
--ind_cs True \
--sample_cs_draw False \
--gen_probs False \
--actv ReLU