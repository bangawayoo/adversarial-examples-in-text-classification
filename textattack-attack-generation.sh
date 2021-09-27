#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
DATASET="imdb"
RECIPE="textfooler pwws bae"
RECIPE="textfooler"
MODEL="cnn-${DATASET} lstm-${DATASET}"
for recipe in $RECIPE
do
  for model in $MODEL
  do
    LOG_FILE_NAME="${model}_${recipe}_neg"
    textattack attack --model $model --num-examples 5000 --checkpoint-interval 2500 --log-to-csv "attack-log/$DATASET/$LOG_FILE_NAME.csv" --model-batch-size 256 --recipe $recipe \
     --num-workers-per-device 16 --filter-by-labels 0\
     2>&1 | tee "attack-log/$DATASET/$LOG_FILE_NAME.txt"

    LOG_FILE_NAME="${model}_${recipe}_pos"
    textattack attack --model $model --num-examples 5000 --checkpoint-interval 2500 --log-to-csv "attack-log/$DATASET/$LOG_FILE_NAME.csv" --model-batch-size 256 --recipe $recipe \
     --num-workers-per-device 16 --filter-by-labels 1\
     2>&1 | tee "attack-log/$DATASET/$LOG_FILE_NAME.txt"
  done
done


#RECIPE="tf-adj"
#for recipe in $RECIPE
#do
#  for model in $MODEL
#  do
#     LOG_FILE_NAME="${model}_${recipe}_neg"
#    textattack attack --model $model --num-examples 5000 --checkpoint-interval 2500 --log-to-csv "attack-log/$DATASET/$LOG_FILE_NAME.csv" --model-batch-size 256 --recipe $recipe \
#     --num-workers-per-device 16 --filter-by-labels 0\ --attack-from-file recipes/textfooler_jin_2019_adjusted.py\
#     2>&1 | tee "attack-log/$DATASET/$LOG_FILE_NAME.txt"
#
#    LOG_FILE_NAME="${model}_${recipe}_pos"
#    textattack attack --model $model --num-examples 5000 --checkpoint-interval 2500 --log-to-csv "attack-log/$DATASET/$LOG_FILE_NAME.csv" --model-batch-size 256 --recipe $recipe \
#     --num-workers-per-device 16 --filter-by-labels 1\ --attack-from-file recipes/textfooler_jin_2019_adjusted.py\
#     2>&1 | tee "attack-log/$DATASET/$LOG_FILE_NAME.txt"
#  done
#done
