#!/bin/bash
cd ccl/

GPU_IDS=(5 6)
MAX_PARALLEL_PER_GPU=1
LOG_DIR="./scripts/ood_nlp/logs"

# Training script
TRAINING_SCRIPT="run_ccl.py"

# Hyperparameter grid
GEN_PROBS_LIST=(True)
CS_DRAW_LIST=(False)
N_BATCH_LIST=(512)

MODEL_SIZE_LIST=('middle' 'small' 'large')

LR_LIST=(3e-4 1e-4 3e-5 1e-5 3e-6)
WD_LIST=(3e-4 1e-4 3e-5 1e-5)

ACTV_PRIOR_LIST=(xtanh lrelu)
ACTIV_LIST=(ReLU)

USE_FC_LIST=(False)
ELBO_ONLY_LIST=(True)

WSUP_LIST=(1)
WGEN_LIST=(1)
WLOGPI_LIST=(1)
WRECON_X_LIST=(1e-1 1 1e+1)
WRECON_E_LIST=(1e-1 1 1e+1)
WRECON_C_LIST=(1e-1 1 1e+1)
WRECON_S_LIST=(1e-1 1 1e+1)

REG_CS_LIST=('mmd')
WREG_CS_LIST=(5e-1)

MU_C_LIST=(1e-1)
MU_S_LIST=(1e-1)
STD_S_LIST=(5e-1 3e-1 1e-1)
STD_X_LIST=(3e-3 5e-2 2e-2 1e-1)
STD_E_LIST=(5e-4 1e-3 3e-2 2e-2)

# early_stop_metric
ES_METRIC_LIST=('total_loss')
RECON_LIST=(False)

IDX=0
MAX_JOBS=$(( ${#GPU_IDS[@]} * MAX_PARALLEL_PER_GPU ))

declare -A GPU_USAGE
for gpu in "${GPU_IDS[@]}"; do
    GPU_USAGE[$gpu]=0
done

declare -A JOB_GPU_MAP

on_child_exit() {
    for pid in "${!JOB_GPU_MAP[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            gpu=${JOB_GPU_MAP[$pid]}
            GPU_USAGE[$gpu]=$(( GPU_USAGE[$gpu] - 1 ))
            unset JOB_GPU_MAP[$pid]
        fi
    done
}

trap 'on_child_exit' CHLD

wait_for_free_slot() {
    while true; do
        RUNNING_JOBS=$(jobs -p | wc -l)
        if [ "$RUNNING_JOBS" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

# Calculate total combinations
TOTAL_COMBINATIONS=$((${#MU_S_LIST[@]} * ${#STD_S_LIST[@]} * ${#ACTIV_LIST[@]} * ${#ACTV_PRIOR_LIST[@]} * \
                     ${#ELBO_ONLY_LIST[@]} * ${#REG_CS_LIST[@]} * ${#WREG_CS_LIST[@]} * ${#WGEN_LIST[@]} * \
                     ${#WLOGPI_LIST[@]} * ${#WRECON_X_LIST[@]} * ${#WRECON_E_LIST[@]} * ${#WRECON_C_LIST[@]} * \
                     ${#WRECON_S_LIST[@]} * ${#WSUP_LIST[@]} * ${#WD_LIST[@]} * ${#LR_LIST[@]} * \
                     ${#ES_METRIC_LIST[@]} * ${#STD_X_LIST[@]} * ${#STD_E_LIST[@]} * ${#MODEL_SIZE_LIST[@]} * \
                     ${#RECON_LIST[@]}))

echo "Total number of combinations: $TOTAL_COMBINATIONS"

# Loop through all combinations of hyperparameters
for GEN_PROBS in "${GEN_PROBS_LIST[@]}"
do
for CS_DRAW in "${CS_DRAW_LIST[@]}"
do
for N_BATCH in "${N_BATCH_LIST[@]}"
do
for USE_FC in "${USE_FC_LIST[@]}"
do
for MU_C in "${MU_C_LIST[@]}"
do
for MU_S in "${MU_S_LIST[@]}"
do
for STD_S in "${STD_S_LIST[@]}"
do
for ACTV in "${ACTIV_LIST[@]}"
do
for ACTV_PRIOR in "${ACTV_PRIOR_LIST[@]}"
do
for ELBO_ONLY in "${ELBO_ONLY_LIST[@]}"
do
for REG_CS in "${REG_CS_LIST[@]}"
do
for WREG_CS in "${WREG_CS_LIST[@]}"
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
for ES_METRIC in "${ES_METRIC_LIST[@]}"
do
for STD_X in "${STD_X_LIST[@]}"
do
for STD_E in "${STD_E_LIST[@]}"
do
for MODEL_SIZE in "${MODEL_SIZE_LIST[@]}"
do
for RECON in "${RECON_LIST[@]}"
do
for WD in "${WD_LIST[@]}"
do
for LR in "${LR_LIST[@]}"
do

LOG_FILE="$LOG_DIR/sweep_ccl_${IDX}.log"

if [ -f "$LOG_FILE" ]; then
    echo "Skipping combination $((IDX + 1)) (already exists: $LOG_FILE)"
    IDX=$((IDX + 1))
    continue
fi

wait_for_free_slot

while true; do
    for gpu in "${GPU_IDS[@]}"; do
        if [ "${GPU_USAGE[$gpu]}" -lt "$MAX_PARALLEL_PER_GPU" ]; then
            GPU_ID=$gpu
            GPU_USAGE[$gpu]=$(( GPU_USAGE[$gpu] + 1 ))
            break 2
        fi
    done
    sleep 1
done

echo "Running combination $((IDX + 1)) on GPU $GPU_ID"

CUDA_VISIBLE_DEVICES=$GPU_ID python $TRAINING_SCRIPT \
--exp_name ood_nlp \
--emb_model gpt \
--model_size $MODEL_SIZE \
--wandb_project CCL_BOSS \
--traindom nli \
--testdoms nli/anli \
--n_bat $N_BATCH \
--n_epk 250 \
--patience 5 \
--lr $LR \
--wl2 $WD \
--early_stop_metric $ES_METRIC \
--recon $RECON \
--wsup $WSUP \
--wgen $WGEN \
--wlogpi $WLOGPI \
--wrecon_x $WRECON_X \
--wrecon_e $WRECON_E \
--wrecon_c $WRECON_C \
--wrecon_s $WRECON_S \
--reg_cs $REG_CS \
--wreg_cs $WREG_CS \
--mu_c $MU_C \
--mu_s $MU_S \
--sig_s $STD_S \
--pstd_x $STD_X \
--pstd_e $STD_E \
--ind_cs True \
--sample_cs_draw $CS_DRAW \
--gen_probs $GEN_PROBS \
--actv $ACTV \
--actv_prior $ACTV_PRIOR \
--elbo_only $ELBO_ONLY \
--y_dtype emb \
--y_emb_option default \
--use_fc $USE_FC \
--no_save True \
--verbose True \
> "$LOG_FILE" 2>&1 &

job_pid=$!
JOB_GPU_MAP[$job_pid]=$GPU_ID

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