#!/bin/bash
cd ../

# List of GPU IDs to use
GPU_IDS=(5 6 7)

# Training script
TRAINING_SCRIPT="run_vae.py"

# Hyperparameter grid
GEN_PROBS_LIST=(True False)
Z_DRAW_LIST=(True False)
N_BATCH_LIST=(128)

LR_LIST=(1e-4 1e-2 1e-6)
WD_LIST=(1e-2 1e-6)

STD_Z_LIST=(0.5)

ELBO_ONLY_LIST=(True False)

WGEN_LIST=(1e-6 1e-4 1e-2)
WSUP_LIST=(1)
WLOGPI_LIST=(1)
WRECON_X_LIST=(1 1e-1)
WRECON_Z_LIST=(1e-2 1e-6)

# Initialize index
IDX=0
MAX_JOBS=3

# Loop through all combinations of hyperparameters
for GEN_PROBS in "${GEN_PROBS_LIST[@]}"
do
for Z_DRAW in "${Z_DRAW_LIST[@]}"
do
for N_BATCH in "${N_BATCH_LIST[@]}"
do
for WD in "${WD_LIST[@]}"
do
for STD_Z in "${STD_Z_LIST[@]}"
do
for ELBO_ONLY in "${ELBO_ONLY_LIST[@]}"
do
for WGEN in "${WGEN_LIST[@]}"
do
for WLOGPI in "${WLOGPI_LIST[@]}"
do
for WRECON_X in "${WRECON_X_LIST[@]}"
do
for WRECON_Z in "${WRECON_Z_LIST[@]}"
do
for WSUP in "${WSUP_LIST[@]}"
do
for LR in "${LR_LIST[@]}"
do

while [ $(jobs -p | wc -l) -ge $MAX_JOBS ]
do
    sleep 1
done

GPU_ID=${GPU_IDS[$((IDX % ${#GPU_IDS[@]}))]}

# Assign GPU and run the script
CUDA_VISIBLE_DEVICES=$GPU_ID python $TRAINING_SCRIPT \
    --exp_name toy \
    --wandb_project CCL_VAE \
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
    --wrecon_z $WRECON_Z \
    --sig_z $STD_Z \
    --sample_z_draw $Z_DRAW \
    --gen_probs $GEN_PROBS \
    --actv ReLU \
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

# Wait for remaining background processes
wait