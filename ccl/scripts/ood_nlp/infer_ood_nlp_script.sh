#! /bin/bash
cd ../../

# Check if required arguments are provided
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <checkpoint> <train_domain> <test_domain> <deploy_id> <deploy_test> <deploy_ood> <gpu_id>"
    echo "Example: $0 002 sa sst5 false true false 0"
    exit 1
fi

# Parse arguments
CKP=$1
TRAINDOM=$2
TESTDOM=$3
DEPLOY_ID=$4
DEPLOY_TEST=$5
DEPLOY_OOD=$6
GPU_ID=${7:-0}  # Default GPU_ID is 0 if not provided

# Common arguments
COMMON_ARGS="--exp_name ood_nlp --emb_model gpt --small_model False --wandb_project CCL_v2 --n_bat 1024 --n_epk 500 --patience 5 --lr 5e-4 --wl2 1e-4 --early_stop_metric total_loss --recon False --wsup 1 --wgen 1 --wlogpi 1 --wrecon_x 1e+2 --wrecon_e 1 --wrecon_c 1 --wrecon_s 1 --reg_cs mmd --wreg_cs 5e-1 --mu_c 1e-1 --mu_s 1e-1 --sig_s 5e-1 --pstd_x 5e-2 --pstd_e 1e-3 --ind_cs True --sample_cs_draw False --gen_probs True --actv ReLU --actv_prior xtanh --elbo_only True --y_dtype emb --y_emb_option default --use_fc False --no_save False --verbose False"

# Run command
CUDA_VISIBLE_DEVICES=$GPU_ID python run_ccl.py $COMMON_ARGS \
    --traindom $TRAINDOM \
    --testdoms $TRAINDOM/$TESTDOM \
    --init_model ckpt_ccl/ood_nlp/ood_nlp_gpt_emb_$CKP.pt \
    --tr_val_split 1 \
    --deploy True \
    --ckp_id $CKP \
    --save_results True \
    --deploy_id $DEPLOY_ID \
    --deploy_test $DEPLOY_TEST \
    --deploy_ood $DEPLOY_OOD