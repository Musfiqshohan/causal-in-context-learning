#! /bin/bash
cd ccl/

# # Val Avg Loss gallant-waterfall-1853
# CUDA_VISIBLE_DEVICES=4 python run_ccl.py --exp_name llm_ret --emb_model gpt --model_size large --wandb_project CCL_LLMR --traindom sentiment --testdoms sentiment/yelp --n_bat 512 --n_epk 250 --patience 5 --lr 1e-4 --wl2 1e-4 --early_stop_metric total_loss --recon False --wsup 1 --wgen 1 --wlogpi 1 --wrecon_x 1e-1 --wrecon_e 1e-1 --wrecon_c 1e-1 --wrecon_s 1e-1 --reg_cs mmd --wreg_cs 5e-1 --mu_c 1e-1 --mu_s 1e-1 --sig_s 5e-1 --pstd_x 3e-3 --pstd_e 5e-4 --ind_cs True --sample_cs_draw False --gen_probs True --actv ReLU --actv_prior xtanh --elbo_only True --y_dtype emb --y_emb_option default --use_fc False \
# --no_save False --verbose True --tr_val_split 0.7 --ckp_id 111 &

# # Val Total Loss smooth-wave-1848
# CUDA_VISIBLE_DEVICES=5 python run_ccl.py --exp_name llm_ret --emb_model gpt --model_size large --wandb_project CCL_LLMR --traindom sentiment --testdoms sentiment/yelp --n_bat 512 --n_epk 250 --patience 5 --lr 1e-4 --wl2 3e-4 --early_stop_metric total_loss --recon False --wsup 1 --wgen 1 --wlogpi 1 --wrecon_x 1e-1 --wrecon_e 1e-1 --wrecon_c 1e-1 --wrecon_s 1e-1 --reg_cs mmd --wreg_cs 5e-1 --mu_c 1e-1 --mu_s 1e-1 --sig_s 5e-1 --pstd_x 3e-3 --pstd_e 5e-4 --ind_cs True --sample_cs_draw False --gen_probs True --actv ReLU --actv_prior xtanh --elbo_only True --y_dtype emb --y_emb_option default --use_fc False \
# --no_save False --verbose True --tr_val_split 0.7 --ckp_id 112 &

# # Current total loss amber-violet-1852
# CUDA_VISIBLE_DEVICES=6 python run_ccl.py --exp_name llm_ret --emb_model gpt --model_size large --wandb_project CCL_LLMR --traindom sentiment --testdoms sentiment/yelp --n_bat 512 --n_epk 250 --patience 5 --lr 3e-4 --wl2 1e-4 --early_stop_metric total_loss --recon False --wsup 1 --wgen 1 --wlogpi 1 --wrecon_x 1e-1 --wrecon_e 1e-1 --wrecon_c 1e-1 --wrecon_s 1e-1 --reg_cs mmd --wreg_cs 5e-1 --mu_c 1e-1 --mu_s 1e-1 --sig_s 5e-1 --pstd_x 3e-3 --pstd_e 5e-4 --ind_cs True --sample_cs_draw False --gen_probs True --actv ReLU --actv_prior xtanh --elbo_only True --y_dtype emb --y_emb_option default --use_fc False \
# --no_save False --verbose True --tr_val_split 0.7 --ckp_id 113 &

# Current total loss 
# CUDA_VISIBLE_DEVICES=3 python run_ccl.py --exp_name llm_ret --emb_model gpt --model_size middle --wandb_project CCL_LLMR --traindom sentiment --testdoms sentiment/yelp --n_bat 512 --n_epk 250 --patience 5 --lr 1e-4 --wl2 5e-4 --early_stop_metric total_loss --recon False --wsup 1 --wgen 1 --wlogpi 1 --wrecon_x 1 --wrecon_e 1 --wrecon_c 1 --wrecon_s 5e-4 --reg_cs mmd --wreg_cs 5e-1 --mu_c 1e-1 --mu_s 1e-1 --sig_s 5e-1 --pstd_x 2e-2 --pstd_e 5e-4 --ind_cs True --sample_cs_draw False --gen_probs True --actv ReLU --actv_prior xtanh --elbo_only True --y_dtype emb --y_emb_option default --use_fc False \
# --no_save False --verbose True --tr_val_split 0.7 --ckp_id 115 &

# # EMB_C_NUM_T_TASK 100% balmy-leaf-2348
# CUDA_VISIBLE_DEVICES=6 python run_ccl.py --exp_name llm_ret --emb_model gpt --model_size large --wandb_project CCL_LLMR --traindom sentiment --testdoms sentiment/yelp --n_bat 512 --n_epk 250 --patience 5 --lr 3e-4 --wl2 3e-5 --early_stop_metric total_loss --recon False --wsup 1 --wgen 1 --wlogpi 1 --wrecon_x 1e-1 --wrecon_e 1e-1 --wrecon_c 1e-1 --wrecon_s 1e-1 --reg_cs mmd --wreg_cs 5e-1 --mu_c 1e-1 --mu_s 1e-1 --sig_s 5e-1 --pstd_x 3e-3 --pstd_e 2e-2 --ind_cs True --sample_cs_draw False --gen_probs True --actv ReLU --actv_prior xtanh --elbo_only True --y_dtype emb --y_emb_option default --use_fc False \
# --no_save False --verbose True --tr_val_split 0.7 --ckp_id 114 &

wait
