#! /bin/bash

# Check if required arguments are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [icl_method] [n_shots] [top_t] [num_points] [use_all] [selection_method] [use_posneg] [ckp] [adapt_id] [gpu_id]"
    echo "task: {task/domain}, sentiment, coreference, nli, etc."
    echo "icl_method: zs (default), icl, ccl, ccl_norm, ccl_adapt, ccl_norm_adapt"
    echo "n_shots: 0 (default), 1, 2, 3, etc."
    echo "use_all: 0 (default) or 1"
    echo "selection_method: knn (default), cossim"
    echo "use_posneg: 0 (default) or 1"
    echo "ckp: 000"
    echo "adapt_id: random (default), val_x_recon, val_e_recon, etc."
    echo "gpu_id: 0 (default), 1, 2, 3, etc."
    exit 1
fi

cd ../

TASK=$1
ICL_METHOD=${2:-"zs"}  # Default to "zs" if not provided
N_SHOTS=${3:-3}        # Default to 3 if not provided
TOP_T=${4:-0}              # Default to 0 if not provided
NUM_POINTS=${5:-50}             # Default to 50 if not provided
USE_ALL=${6:-1}        # Default to 0 if not provided
RM=${7:-"knn"}      # Default to "cossim" if not provided
USE_PN=${8:-0}        # Default to 0 if not provided
CKP=${9:-"413"}        # Default to "000" if not provided
ADAPT_ID=${10:-"random"} # Default to "random" if not provided
GPU_ID=${11:-1}         # Default to 1 if not provided

# Base command with common parameters
BASE_CMD="python prompt_generation.py --gpu_id $GPU_ID --exp_name ood_nlp --domain_name ood --emb_model gpt --prompt_style 0 --use_system_prompt --use_instruction --x_title Text --ckp $CKP --selection_method $RM --overwrite"


# Set num_shots based on ICL method
if [ "$ICL_METHOD" = "zs" ]; then
    NUM_SHOTS=0
else
    NUM_SHOTS=$N_SHOTS
fi

# Handle different ICL methods
case $ICL_METHOD in
    "zs")
        $BASE_CMD --task_name $TASK --icl_method zs --num_shots 0 --use_all
        ;;
    "icl")
        if [ "$USE_ALL" = "1" ]; then
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method icl --num_shots $NUM_SHOTS --use_all --use_posneg --num_points $NUM_POINTS
            else
                $BASE_CMD --task_name $TASK --icl_method icl --num_shots $NUM_SHOTS --use_all --num_points $NUM_POINTS
            fi
        else
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method icl --num_shots $NUM_SHOTS --use_posneg --num_points $NUM_POINTS
            else
                $BASE_CMD --task_name $TASK --icl_method icl --num_shots $NUM_SHOTS --num_points $NUM_POINTS
            fi
        fi
        ;;
    "ccl")
        if [ "$USE_ALL" = "1" ]; then
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $NUM_POINTS --top_t $TOP_T --use_all --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $NUM_POINTS --top_t $TOP_T --use_all
            fi
        else
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $NUM_POINTS --top_t $TOP_T --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $NUM_POINTS --top_t $TOP_T
            fi
        fi
        ;;
    "ccl_norm")
        if [ "$USE_ALL" = "1" ]; then
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $NUM_POINTS --top_t $TOP_T --norm --use_all --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $NUM_POINTS --top_t $TOP_T --norm --use_all
            fi
        else
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $NUM_POINTS --top_t $TOP_T --norm --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $NUM_POINTS --top_t $TOP_T --norm
            fi
        fi
        ;;
    "ccl_r")
        if [ "$USE_ALL" = "1" ]; then
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --use_all --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --use_all
            fi
        else
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K
            fi
        fi
        ;;
    "ccl_norm_r")
        if [ "$USE_ALL" = "1" ]; then
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --use_all --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --use_all
            fi
        else
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm
            fi
        fi
        ;;
    "ccl_adapt")
        if [ "$USE_ALL" = "1" ]; then
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $P --top_t $K --adapt_ood --adapt_id $ADAPT_ID --use_all --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $P --top_t $K --adapt_ood --adapt_id $ADAPT_ID --use_all
            fi
        else
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $P --top_t $K --adapt_ood --adapt_id $ADAPT_ID --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $P --top_t $K --adapt_ood --adapt_id $ADAPT_ID
            fi
        fi
        ;;
    "ccl_adapt_r")
        if [ "$USE_ALL" = "1" ]; then
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --adapt_ood --adapt_id $ADAPT_ID --use_all --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --adapt_ood --adapt_id $ADAPT_ID --use_all
            fi
        else
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --adapt_ood --adapt_id $ADAPT_ID --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --adapt_ood --adapt_id $ADAPT_ID
            fi
        fi
        ;;
    "ccl_norm_adapt")
        if [ "$USE_ALL" = "1" ]; then
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --adapt_ood --adapt_id $ADAPT_ID --use_all --use_posneg --num_points $NUM_POINTS
            else
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --adapt_ood --adapt_id $ADAPT_ID --use_all --num_points $NUM_POINTS
            fi
        else
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --adapt_ood --adapt_id $ADAPT_ID --use_posneg --num_points $NUM_POINTS
            else
                $BASE_CMD --task_name $TASK --icl_method ccl --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --adapt_ood --adapt_id $ADAPT_ID --num_points $NUM_POINTS
            fi
        fi
        ;;
    "ccl_norm_adapt_r")
        if [ "$USE_ALL" = "1" ]; then
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --adapt_ood --adapt_id $ADAPT_ID --use_all --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --adapt_ood --adapt_id $ADAPT_ID --use_all
            fi
        else
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --adapt_ood --adapt_id $ADAPT_ID --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl_r --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --adapt_ood --adapt_id $ADAPT_ID
            fi
        fi
        ;;
    "ccl_norm_adapt_s")
        if [ "$USE_ALL" = "1" ]; then
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl_s --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --adapt_ood --adapt_id $ADAPT_ID --use_all --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl_s --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --adapt_ood --adapt_id $ADAPT_ID --use_all
            fi
        else
            if [ "$USE_PN" = "1" ]; then
                $BASE_CMD --task_name $TASK --icl_method ccl_s --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --adapt_ood --adapt_id $ADAPT_ID --use_posneg
            else
                $BASE_CMD --task_name $TASK --icl_method ccl_s --num_shots $NUM_SHOTS --num_points $P --top_t $K --norm --adapt_ood --adapt_id $ADAPT_ID
            fi
        fi
        ;;
    *)
        echo "Invalid ICL method. Please use: zs, icl, ccl, ccl_norm, ccl_adapt, ccl_norm_adapt"
        exit 1
        ;;
esac

# bash prompt_gen_script_ood_nlp.sh {task/domain} zs 0

# bash prompt_gen_script_ood_nlp.sh {task/domain} icl 0
# bash prompt_gen_script_ood_nlp.sh {task/domain} icl 1
# bash prompt_gen_script_ood_nlp.sh {task/domain} ccl_norm 0
# bash prompt_gen_script_ood_nlp.sh {task/domain} ccl_norm 1
# bash prompt_gen_script_ood_nlp.sh {task/domain} ccl_norm_adapt 0
# bash prompt_gen_script_ood_nlp.sh {task/domain} ccl_norm_adapt 1