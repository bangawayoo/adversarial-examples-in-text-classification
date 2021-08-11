#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
#DATASET="imdb"
#RECIPE="pwws textfooler"
##RECIPE="pwws"
#MODEL="roberta-base-${DATASET}"
#for recipe in $RECIPE
#do
#  for model in $MODEL
#  do
#    LOG_FILE_NAME="${model}_${recipe}"
#    textattack attack --model $model --num-examples 10000 --log-to-csv "attack-log/$LOG_FILE_NAME.csv" --model-batch-size 64 --recipe $recipe \
#     --num-workers-per-device 4 --checkpoint-interval 500 --checkpoint-dir "attack-log/checkpoint" \
#     2>&1 | tee "attack-log/$LOG_FILE_NAME.txt"
#  done
#done



DATASET="imdb"
RECIPE="tf-adj"
MODEL="bert-base-uncased-${DATASET}"
#MODEL="bert-base-uncased-${DATASET}"
for recipe in $RECIPE
do
  for model in $MODEL
  do
     LOG_FILE_NAME="${model}_${recipe}"
    textattack attack --model $model --num-examples 10000 --attack-from-file recipes/textfooler_jin_2019_adjusted.py \
    --log-to-csv "attack-log/$LOG_FILE_NAME.csv" --model-batch-size 64\
     --num-workers-per-device 8 --checkpoint-interval 500 --checkpoint-dir "attack-log/checkpoint" \
     2>&1 | tee "attack-log/$LOG_FILE_NAME.txt"
  done
done
