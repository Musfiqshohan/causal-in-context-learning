#! /bin/bash
# bash ccl/scripts/ood_nlp/run_infer_ood_nlp.sh

# CKP task domain deploy_id deploy_ood gpu_id
cd ccl/scripts/ood_nlp/

for CKP in 001 002 003 004 005 006
do
# echo "Infer ID Only"
bash infer_ood_nlp_script.sh $CKP nli anli True False False 0 &
done
wait

for CKP in 001 002 003 004 005 006
do
bash infer_ood_nlp_script.sh $CKP sa amazon False True False 0 &
bash infer_ood_nlp_script.sh $CKP nli mnli False True False 1 &
bash infer_ood_nlp_script.sh $CKP td civil_comments False True False 2 &
done

wait

for CKP in 001 002 003 004 005 006
do
bash infer_ood_nlp_script.sh $CKP sa sst5 False False True 0 &
bash infer_ood_nlp_script.sh $CKP sa semeval False False True 0 &
bash infer_ood_nlp_script.sh $CKP sa dynasent False False True 0 &

bash infer_ood_nlp_script.sh $CKP nli anli False False True 1 &
bash infer_ood_nlp_script.sh $CKP nli contract_nli False False True 1 &
bash infer_ood_nlp_script.sh $CKP nli wanli False False True 1 &

bash infer_ood_nlp_script.sh $CKP td adv_civil False False True 2 &
bash infer_ood_nlp_script.sh $CKP td implicit_hate False False True 2 &
bash infer_ood_nlp_script.sh $CKP td toxigen False False True 2 &

wait
done