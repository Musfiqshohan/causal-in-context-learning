#! /bin/bash
cd ../../

# Check if required arguments are provided
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <gpu_id> <checkpoint> <train_domain> <test_domain> <deploy_ood> <batch_size> <epochs> <wreg_cs> <wada_x> <wada_e> <lam_c> <lam_s> <lr_c> <lr_s> <wd_c> <wd_s>"
    echo "Example: $0 0 002 sa sst5 False 512 100 5e-1 1 1 1e-4 1e-4 1e-3 1e-6 1e-4 1e-3"
    exit 1
fi

# Parse arguments
GPU_ID=${1:-0}  # Default GPU_ID is 0 if not provided
CKP=$2
TRAINDOM=$3
TESTDOM=$4
DEPLOY_OOD=$5
BATCH_SIZE=$6
EPOCHS=$7
WREG_CS=${8}
WADA_X=${9}
WADA_E=${10}
LAM_C=${11}
LAM_S=${12}
LR_C=${13}
LR_S=${14}
WD_C=${15}
WD_S=${16}


# Common arguments
COMMON_ARGS="--exp_name llm_ret --emb_model gpt --wandb_project CCL_v2 --patience 5 --recon False --reg_cs mmd"

# Run command
CUDA_VISIBLE_DEVICES=$GPU_ID python run_ccl.py $COMMON_ARGS \
--traindom $TRAINDOM \
--testdoms $TRAINDOM/$TESTDOM \
--n_bat $BATCH_SIZE \
--n_epk $EPOCHS \
--init_model ckpt_ccl/llm_ret/llm_ret_gpt_emb_$CKP.pt \
--ckp_id $CKP \
--deploy True \
--adapt_ood True \
--online_adapt False \
--latent_update True \
--wreg_cs $WREG_CS \
--wada_x $WADA_X \
--wada_e $WADA_E \
--lam_c $LAM_C \
--lam_s $LAM_S \
--lr_l_c $LR_C \
--lr_l_s $LR_S \
--wd_l_c $WD_C \
--wd_l_s $WD_S \
--save_results False \
--deploy_id False \
--deploy_test False \
--deploy_ood $DEPLOY_OOD