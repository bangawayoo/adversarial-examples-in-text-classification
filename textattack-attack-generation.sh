#!/bin/bash

DATASET="imdb"
RECIPE="clare"
MODEL="roberta-base-${DATASET} bert-base-uncased-${DATASET}"
for recipe in $RECIPE
do
  for model in $MODEL
  do
    LOG_FILE_NAME="${model}_${recipe}.txt"
    textattack attack --recipe $recipe --model $model --num-examples 2000 --log-to-csv ./attack-log/ --model-batch-size 64 --parallel 2>&1 | tee attack-log/$LOG_FILE_NAME
  done
done

