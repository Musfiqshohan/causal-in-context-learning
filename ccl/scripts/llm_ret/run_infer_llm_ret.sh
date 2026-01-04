#! /bin/bash
# Arguments:
# CKP Task Domain IS_Deploy_ID_train IS_Deploy_ID_test IS_Deploy_OOD GPU_ID

cd ccl/scripts/llm_ret/

# ID Train
bash infer_llm_ret_script.sh $1 commonsense piqa True False False 0 &
wait

# ID Test
# bash infer_llm_ret_script.sh $1 commonsense copa False True False 0 &
# bash infer_llm_ret_script.sh $1 coreference wsc False True False 1 &
# bash infer_llm_ret_script.sh $1 nli rte False True False 2 &
# bash infer_llm_ret_script.sh $1 sentiment sentiment140 False True False 3 &
wait

# OOD
bash infer_llm_ret_script.sh $1 commonsense piqa False False True 0 &
bash infer_llm_ret_script.sh $1 coreference wsc273 False False True 1 &
bash infer_llm_ret_script.sh $1 nli qnli False False True 2 &
bash infer_llm_ret_script.sh $1 sentiment yelp False False True 3 &
wait

# Done
# bash ccl/scripts/llm_ret/run_infer_llm_ret.sh 111
# bash ccl/scripts/llm_ret/run_infer_llm_ret.sh 100
# bash ccl/scripts/llm_ret/run_infer_llm_ret.sh 208