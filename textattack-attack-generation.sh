#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
DATASET="ag-news"
RECIPE="textfooler"
MODEL="roberta-base-${DATASET}"
for recipe in $RECIPE
do
  for model in $MODEL
  do
    LOG_FILE_NAME="${model}_${recipe}"
    textattack attack --model $model --num-examples 10000 --log-to-csv "attack-log/ag-news/$LOG_FILE_NAME.csv" --model-batch-size 128 --recipe $recipe \
     --num-workers-per-device 16 --checkpoint-interval 1000 --checkpoint-dir "attack-log/checkpoint" \
     2>&1 | tee "attack-log/ag-news/$LOG_FILE_NAME.txt"
  done
done



#DATASET="ag-news"
#RECIPE="tf-adj"
#MODEL="roberta-base-${DATASET}"
#for recipe in $RECIPE
#do
#  for model in $MODEL
#  do
#     LOG_FILE_NAME="${model}_${recipe}"
#    textattack attack --model $model --num-examples 10000 --attack-from-file recipes/textfooler_jin_2019_adjusted.py \
#    --log-to-csv "attack-log/ag-news/$LOG_FILE_NAME.csv" --model-batch-size 64\
#     --num-workers-per-device 8 --checkpoint-interval 500 --checkpoint-dir "attack-log/checkpoint" \
#     2>&1 | tee "attack-log/ag-news/$LOG_FILE_NAME.txt"
#  done
#done
