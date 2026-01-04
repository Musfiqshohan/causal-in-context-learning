#! /bin/bash
# CKP task domain deploy_id deploy_ood gpu_id

for CKP in 300 512
do
bash infer_mgsm_script.sh $CKP mgsm mgsm True False 5 &
bash infer_mgsm_script.sh $CKP mgsm mgsm_7 False True 5 &
bash infer_mgsm_script.sh $CKP mgsm mgsm_8 False True 5 &
bash infer_mgsm_script.sh $CKP mgsm mgsm_9 False True 5 &
bash infer_mgsm_script.sh $CKP mgsm mgsm_10 False True 5 &
done

wait