#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
DATASET="sst2"
RECIPE="textfooler pwws bae"
RECIPE="textfooler"
MODEL="cnn-${DATASET} lstm-${DATASET}"
#for recipe in $RECIPE
#do
#  for model in $MODEL
#  do
#    LOG_FILE_NAME="${model}_${recipe}"
#    textattack attack --model $model --num-examples -1 --log-to-csv "attack-log/$DATASET/$LOG_FILE_NAME.csv" --model-batch-size 256 --recipe $recipe \
#     --num-workers-per-device 16 --dataset-from-file sst2_dataset.py\
#     2>&1 | tee "attack-log/$DATASET/$LOG_FILE_NAME.txt"
#  done
#done


RECIPE="tf-adj"
for recipe in $RECIPE
do
  for model in $MODEL
  do
     LOG_FILE_NAME="${model}_${recipe}"
    textattack attack --model $model --num-examples -1 --log-to-csv "attack-log/$DATASET/$LOG_FILE_NAME.csv" --model-batch-size 256\
     --num-workers-per-device 16 --dataset-from-file sst2_dataset.py\
     --attack-from-file recipes/textfooler_jin_2019_adjusted.py\
     2>&1 | tee "attack-log/$DATASET/$LOG_FILE_NAME.txt"

  done
done
