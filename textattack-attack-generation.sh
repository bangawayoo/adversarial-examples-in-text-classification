#!/bin/bash

DATASET="imdb"
RECIPE="alzantot"
MODEL="roberta-base-${DATASET} bert-base-uncased-${DATASET}"
for recipe in $RECIPE
do
  for model in $MODEL
  do
    LOG_FILE_NAME="${model}_${recipe}.txt"
    textattack attack --recipe $recipe --model $model --num-examples 2000 --log-to-csv ./attack_log/ --model-batch-size 64 2>&1 | tee attack_log/$LOG_FILE_NAME
  done
done

