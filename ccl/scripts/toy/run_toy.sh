#!/bin/bash
cd ../

# List of GPU IDs to use
GPU_IDS=(1 2 3 4 5 6 7)

# Training script
TRAINING_SCRIPT="run_ccl.py"

# Hyperparameter grid
GEN_PROBS_LIST=(True False)
CS_DRAW_LIST=(True False)
N_BATCH_LIST=(128 512)

LR_LIST=(1e-4 1e-2 1e-6)
WD_LIST=(1e-2 1e-6)

STD_S_LIST=(0.5 1 0.01)
ACTV_PRIOR_LIST=(xtanh)

ELBO_ONLY_LIST=(True False)

WGEN_LIST=(1e-6 1e-4 1e-2)
WSUP_LIST=(1)
WLOGPI_LIST=(1)
WRECON_X_LIST=(1 1e-1)
WRECON_E_LIST=(1 1e-1)
WRECON_C_LIST=(1e-2 1e-6)
WRECON_S_LIST=(1e-2 1e-6)

# Initialize index
IDX=0
MAX_JOBS=14

# Loop through all combinations of hyperparameters
for GEN_PROBS in "${GEN_PROBS_LIST[@]}"
do
for CS_DRAW in "${CS_DRAW_LIST[@]}"
do
for N_BATCH in "${N_BATCH_LIST[@]}"
do
for LR in "${LR_LIST[@]}"
do
for WD in "${WD_LIST[@]}"
do
for STD_S in "${STD_S_LIST[@]}"
do
for ACTV_PRIOR in "${ACTV_PRIOR_LIST[@]}"
do
for ELBO_ONLY in "${ELBO_ONLY_LIST[@]}"
do
for WGEN in "${WGEN_LIST[@]}"
do
for WLOGPI in "${WLOGPI_LIST[@]}"
do
for WRECON_X in "${WRECON_X_LIST[@]}"
do
for WRECON_E in "${WRECON_E_LIST[@]}"
do
for WRECON_C in "${WRECON_C_LIST[@]}"
do
for WRECON_S in "${WRECON_S_LIST[@]}"
do
for WSUP in "${WSUP_LIST[@]}"
do

while [ $(jobs -p | wc -l) -ge $MAX_JOBS ]
do
    sleep 1
done

GPU_ID=${GPU_IDS[$((IDX % ${#GPU_IDS[@]}))]}

# Assign GPU and run the script
CUDA_VISIBLE_DEVICES=$GPU_ID python $TRAINING_SCRIPT \
    --exp_name toy \
    --wandb_project CCL_v2 \
    --traindom toy \
    --testdoms toy \
    --n_bat $N_BATCH \
    --n_epk 500 \
    --lr $LR \
    --wl2 $WD \
    --wsup $WSUP \
    --wgen $WGEN \
    --wlogpi $WLOGPI \
    --wrecon_x $WRECON_X \
    --wrecon_e $WRECON_E \
    --wrecon_c $WRECON_C \
    --wrecon_s $WRECON_S \
    --sig_c 1. \
    --sig_s $STD_S \
    --ind_cs True \
    --sample_cs_draw $CS_DRAW \
    --gen_probs $GEN_PROBS \
    --actv ReLU \
    --actv_prior $ACTV_PRIOR \
    --elbo_only $ELBO_ONLY \
    --no_save True \
    > /dev/null 2>&1 &

    IDX=$((IDX + 1))

done
done
done
done
done
done
done
done
done
done
done
done
done
done
done

# Wait for remaining background processes
wait