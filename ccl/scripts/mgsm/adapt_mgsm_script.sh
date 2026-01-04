#! /bin/bash
cd ../

# Check if required arguments are provided
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <gpu_id> <checkpoint> <train_domain> <test_domain> <deploy_ood> <online_adapt> <batch_size> <epochs> <wreg_cs> <wada_x> <wada_e> <lam_c> <lam_s> <lr_discr> <lr_gen> <wl2_discr> <wl2_gen>"
    echo "Example: $0 0 002 sa sst5 False True 512 100 5e-1 1 1 1e-4 1e-4 1e-3 1e-6 1e-4"
    exit 1
fi

# Parse arguments
GPU_ID=${1:-0}  # Default GPU_ID is 0 if not provided
CKP=$2
TRAINDOM=$3
TESTDOM=$4
DEPLOY_OOD=$5
ONLINE_ADAPT=$6
BATCH_SIZE=$7
EPOCHS=$8
WREG_CS=${9}
WADA_X=${10}
WADA_E=${11}
LAM_C=${12}
LAM_S=${13}
LR_DISCR=${14}
LR_GEN=${15}
WL2_DISCR=${16}
WL2_GEN=${17}

# Common arguments
COMMON_ARGS="--exp_name mgsm --emb_model gpt --wandb_project CCL_v2 --patience 5"

# Run command
CUDA_VISIBLE_DEVICES=$GPU_ID python run_ccl.py $COMMON_ARGS \
--traindom $TRAINDOM \
--testdoms $TESTDOM \
--n_bat $BATCH_SIZE \
--n_epk $EPOCHS \
--init_model ckpt_ccl/mgsm/mgsm_gpt_eq_$CKP.pt \
--deploy True \
--adapt_ood True \
--online_adapt $ONLINE_ADAPT \
--wreg_cs $WREG_CS \
--wada_x $WADA_X \
--wada_e $WADA_E \
--lam_c $LAM_C \
--lam_s $LAM_S \
--lr_discr $LR_DISCR \
--lr_gen $LR_GEN \
--wl2_discr $WL2_DISCR \
--wl2_gen $WL2_GEN \
--save_results False \
--deploy_ood $DEPLOY_OOD