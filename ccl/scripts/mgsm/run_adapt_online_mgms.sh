#!/bin/bash

# Available GPUs
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
MAX_JOBS_PER_GPU=2 # Maximum number of jobs per GPU
declare -A RUNNING_JOBS # Track running jobs per GPU

BS_ARR=(32)
EPOCHS_ARR=(100)
W_CS_ARR=(1 0.1)
W_X_ARR=(0.1 1)
W_E_ARR=(0.1 1)
LAM_C_ARR=(1 0.1)
LAM_S_ARR=(1 0.1)
LR_DISCR_ARR=(1e-5 1e-4 5e-6)
WL2_DISCR_ARR=(1e-3 1e-4 1e-5)
LR_GEN_ARR=(1e-1)
WL2_GEN_ARR=(1e-1)

CHECKPOINT=300
DEPLOY_OOD=True
ONLINE_ADAPT=True

TRAIN_DOMAINS=("mgsm")
TEST_DOMAINS=("mgsm_7", "mgsm_8" "mgsm_9", "mgsm_10")

# Function to get least loaded GPU
get_available_gpu() {
    local min_jobs=999
    local selected_gpu=-1

    for gpu in "${GPUS[@]}"; do
        local current_jobs=${RUNNING_JOBS[$gpu]:-0}
        if (( current_jobs < min_jobs )) && (( current_jobs < MAX_JOBS_PER_GPU )); then
            min_jobs=$current_jobs
            selected_gpu=$gpu
        fi
    done

    echo $selected_gpu
}

# Function to wait for any job to finish
wait_for_job() {
    while :; do
        for gpu in "${GPUS[@]}"; do
            if (( RUNNING_JOBS[$gpu] > 0 )); then
                wait -n
                RUNNING_JOBS[$gpu]=$((${RUNNING_JOBS[$gpu]} - 1))
                return
            fi
        done
        sleep 1
    done
}

# Function to update job count with locking
start_job() {
    local gpu=$1
    RUNNING_JOBS[$gpu]=$((${RUNNING_JOBS[$gpu]:-0} + 1))
}

for BS in "${BS_ARR[@]}"; do
  for EPOCHS in "${EPOCHS_ARR[@]}"; do
    for WCS in "${W_CS_ARR[@]}"; do
      for WX in "${W_X_ARR[@]}"; do
        for WE in "${W_E_ARR[@]}"; do
          for LAMC in "${LAM_C_ARR[@]}"; do
            for LAMS in "${LAM_S_ARR[@]}"; do
              for LRDISCR in "${LR_DISCR_ARR[@]}"; do
                for LRGEN in "${LR_GEN_ARR[@]}"; do
                  for WL2D in "${WL2_DISCR_ARR[@]}"; do
                    for WL2G in "${WL2_GEN_ARR[@]}"; do
                      for TRAIN_DOMAIN in "${TRAIN_DOMAINS[@]}"; do
                        TEST_DOMAINS=("${TEST_DOMAINS[@]}")
                        echo ""
                        exit 1

                        echo ""
                        echo "==== [TRAIN_DOMAIN: $TRAIN_DOMAIN] ===="
                        for TEST_DOMAIN in "${TEST_DOMAINS[@]}"; do
                          while :; do
                            GPU_ID=$(get_available_gpu)
                            if [ $GPU_ID -ne -1 ]; then
                              start_job $GPU_ID

                              echo "Running on GPU $GPU_ID: CKP=$CHECKPOINT / train=$TRAIN_DOMAIN / test=$TEST_DOMAIN / Deploy OOD=$DEPLOY_OOD / Online Adapt=$ONLINE_ADAPT / BS=$BS / EPOCHS=$EPOCHS / WCS=$WCS / WX=$WX / WE=$WE / LAMC=$LAMC / LAMS=$LAMS / LRDISCR=$LRDISCR / LRGEN=$LRGEN / WL2D=$WL2D / WL2G=$WL2G"
                              (
                                bash adapt_mgsm_script.sh $GPU_ID $CHECKPOINT $TRAIN_DOMAIN $TEST_DOMAIN $DEPLOY_OOD $ONLINE_ADAPT $BS $EPOCHS $WCS $WX $WE $LAMC $LAMS $LRDISCR $LRGEN $WL2D $WL2G
                                if [ $? -ne 0 ]; then
                                    echo "[ERROR] Script failed on GPU $GPU_ID"
                                    exit 1
                                fi
                                RUNNING_JOBS[$GPU_ID]=$((${RUNNING_JOBS[$GPU_ID]} - 1))
                              ) &
                              break
                            else
                              wait_for_job
                            fi
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

# Wait for all jobs to finish
wait
