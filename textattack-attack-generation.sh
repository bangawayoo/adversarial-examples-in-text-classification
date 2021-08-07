#!/bin/bash

DATASET="imdb"
RECIPE="tf-adj"
MODEL="bert-base-uncased-${DATASET} roberta-base-${DATASET}"
for recipe in $RECIPE
do
  for model in $MODEL
  do
    LOG_FILE_NAME="${model}_${recipe}.txt"
    textattack attack --attack-from-file recipes/textfooler_jin_2019_adjusted.py --model $model --num-examples 25000 --log-to-csv ./attack-log/$LOG_FILE_NAME.csv --model-batch-size 64 --parallel 2>&1 | tee attack-log/$LOG_FILE_NAME
  done
done

