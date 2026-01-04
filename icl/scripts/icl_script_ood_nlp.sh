#!/bin/bash

cd ../

# Default values for arguments
LLM_MODEL='gpt'
EMB_MODEL='gpt'
default_tasks=("sentiment/yelp" "nli/qnli" "coreference/wsc273" "commonsense/piqa")
default_methods=("ccl") # Add more methods if needed
default_n_shots=8        # Default value for --num_shots
default_n_subset=1000    # Default value for --n_subset
default_batch_size=1
default_RM=("cossim")
use_all=false
adapt_ood=false
norm=false
overwrite=false
top_t=1
num_points=50

# Parse input arguments
while getopts "g:b:e:t:m:c:s:k:p:l:r:z:uano" opt; do
  case $opt in
    g) LLM_MODEL=$OPTARG ;;
    b) batch_size=$OPTARG ;;                      # Number of shots
    e) EMB_MODEL=$OPTARG ;;
    t) IFS=',' read -r -a tasks <<< "$OPTARG" ;;   # List of tasks, split by comma
    m) IFS=',' read -r -a methods <<< "$OPTARG" ;; # List of methods, split by comma
    s) n_subset=$OPTARG ;;                         # Number of subsets
    l) n_shots=$OPTARG ;;                          # Number of shots
    z) adapt_id=$OPTARG ;;                         # Adapt ID
    u) use_all=true ;;                             # Use all options
    a) adapt_ood=true ;;                           # Enable --adapt_ood
    n) norm=true ;;                                # Enable --norm
    o) overwrite=true ;;                           # Enable --overwrite
    c) ckp=$OPTARG ;;
    r) RM=$OPTARG ;;
    k) top_t=$OPTARG ;;
    p) num_points=$OPTARG ;;
    *) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

# Fallback to default values if no arguments are provided
tasks=${tasks[@]:-${default_tasks[@]}}
methods=${methods[@]:-${default_methods[@]}}
n_shots=${n_shots:-$default_n_shots}
n_subset=${n_subset:-$default_n_subset}
n_batch=${n_batch:-$default_n_batch}
ckp=${ckp:-"000"}
RM=${RM:-$default_RM}

# Prepare additional flags
use_all_flag=""
if $use_all; then
    use_all_flag="--use_all"
fi

adapt_ood_flag=""
adapt_id_flag=""
if $adapt_ood; then
    adapt_ood_flag="--adapt_ood"
    adapt_id_flag="--adapt_id $adapt_id"
fi

norm_flag=""
if $norm; then
    norm_flag="--norm"
fi

overwrite_flag=""
if $overwrite; then
    overwrite_flag="--overwrite"
fi

# Loop through all combinations
for task in "${tasks[@]}"; do
    for method in "${methods[@]}"; do
        if [ "$method" = "zs" ] || [ "$method" = "fix" ]; then
            # Zero-shot case (num_shots=0)
            python run_icl.py --seed 0 --exp_name ood_nlp \
                --task_name "$task" \
                --domain_name ood \
                --batch_size $batch_size \
                --n_subset $n_subset \
                --emb_model $EMB_MODEL \
                --icl_method "$method" \
                --num_shots 0 \
                --llm_model $LLM_MODEL \
                --ckp "$ckp" \
                $use_all_flag \
                $overwrite_flag
        else
            # Other methods (num_shots=8)
            if [ "$method" = "ccl" ]; then
                echo "python run_icl.py --seed 0 --exp_name ood_nlp --task_name $task --domain_name ood --batch_size $batch_size --n_subset $n_subset --emb_model $EMB_MODEL --icl_method $method --num_shots $n_shots $norm_flag --llm_model $LLM_MODEL $use_all_flag $adapt_ood_flag $adapt_id_flag --ckp $ckp --selection_method $RM --top_t $top_t --num_points $num_points $overwrite_flag"

                python run_icl.py --seed 0 --exp_name ood_nlp \
                            --task_name "$task" \
                            --domain_name ood \
                            --batch_size $batch_size \
                            --n_subset $n_subset \
                            --emb_model $EMB_MODEL \
                            --icl_method "$method" \
                            --num_shots $n_shots \
                            $norm_flag \
                            --llm_model $LLM_MODEL \
                            $use_all_flag \
                            $adapt_ood_flag \
                            $adapt_id_flag \
                            --ckp "$ckp" \
                            --selection_method "$RM" \
                            --top_t $top_t \
                            --num_points $num_points \
                            $overwrite_flag
            else
                python run_icl.py --seed 0 --exp_name ood_nlp \
                            --task_name "$task" \
                            --domain_name ood \
                            --batch_size $batch_size \
                            --n_subset $n_subset \
                            --emb_model $EMB_MODEL \
                            --icl_method "$method" \
                            --num_shots $n_shots \
                            $norm_flag \
                            --llm_model $LLM_MODEL \
                            $use_all_flag \
                            $adapt_ood_flag \
                            $adapt_id_flag \
                            --ckp "$ckp" \
                            --selection_method "$RM" \
                            --top_t $top_t \
                            --num_points $num_points \
                            $overwrite_flag
            fi
        fi
    done
done

wait  # Wait for all background processes to complete