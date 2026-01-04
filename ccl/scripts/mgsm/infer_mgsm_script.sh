#! /bin/bash
cd ../

# Check if required arguments are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <checkpoint> <train_domain> <test_domain> <deploy_ood> <gpu_id>"
    echo "Example: $0 002 sa sst5 false 0"
    exit 1
fi

# Parse arguments
CKP=$1
TRAINDOM=$2
TESTDOM=$3
DEPLOY_ID=$4
DEPLOY_OOD=$5
GPU_ID=${6:-0}  # Default GPU_ID is 0 if not provided

# Common arguments
COMMON_ARGS="--exp_name ood_nlp --emb_model gpt --small_model False --wandb_project CCL_v2 --n_bat 1024 --n_epk 500 --patience 5 --no_save False --verbose False"

# Run command
CUDA_VISIBLE_DEVICES=$GPU_ID python run_ccl.py $COMMON_ARGS \
    --traindom $TRAINDOM \
    --testdoms $TESTDOM \
    --init_model ckpt_ccl/mgsm/mgsm_gpt_emb_$CKP.pt \
    --tr_val_split 1 \
    --deploy True \
    --save_results True \
    --deploy_id $DEPLOY_ID \
    --deploy_ood $DEPLOY_OOD