#! /bin/bash

# Plz check if no_save False False, and set the ckp_id &
cd ../

CUDA_VISIBLE_DEVICES=1 python run_ccl.py run_ccl.py --exp_name mgsm --emb_model gpt --small_model False --wandb_project CCL_gpt_opt_4 --traindom mgsm --testdoms mgsm --n_bat 256 --n_epk 500 --patience 5 --lr 1e-3 --wl2 1e-3 --early_stop_metric total_loss --recon False --wsup 1 --wgen 1 --wlogpi 1 --wrecon_x 1e+2 --wrecon_e 1e+2 --wrecon_c 1 --wrecon_s 1 --reg_cs mmd --wreg_cs 1 --mu_c 1e-1 --mu_s 1e-1 --sig_s 5e-1 --pstd_x 5e-2 --pstd_e 1e-3 --ind_cs True --sample_cs_draw False --gen_probs True --actv ReLU --actv_prior lrelu --elbo_only True --y_dtype emb --y_emb_option eq --use_fc False --no_save False --verbose True &

wait