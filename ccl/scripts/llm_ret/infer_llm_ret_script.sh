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
COMMON_ARGS="--exp_name llm_ret --emb_model intfloat --small_model False --wandb_project CCL_v2 --n_bat 1024 --y_dtype emb --y_emb_option default --use_fc False --no_save False --verbose False"

# Run command
CUDA_VISIBLE_DEVICES=$GPU_ID python run_ccl.py $COMMON_ARGS \
    --traindom $TRAINDOM \
    --testdoms $TRAINDOM/$TESTDOM \
    --init_model ckpt_ccl/llm_ret/llm_ret_intfloat_emb_$CKP.pt \
    --tr_val_split 1 \
    --deploy True \
    --ckp_id $CKP \
    --save_results True \
    --deploy_id $DEPLOY_ID \
    --deploy_test $DEPLOY_TEST \
    --deploy_ood $DEPLOY_OOD