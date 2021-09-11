export CUDA_VISIBLE_DEVICES=0,1
DATASET="ag-news"
MODEL=("bert" "roberta")
TARGET_MODEL=("textattack/bert-base-uncased-$DATASET" "textattack/roberta-base-$DATASET")
RECIPE="pwws textfooler bae tf-adj"
EXP_NAME="tmp"
PARAM_PATH="params/attention_key-include.json"
SEED_START=0
SEED_END=0
GPU=0

python utils/dataset.py $DATASET

for ((i=0; i< ${#MODEL[@]}; i++ ));
do
  for recipe in $RECIPE
  do
    for seed in $(seq $SEED_START $SEED_END)
    do
      python main.py --dataset $DATASET\
      --test_adv attack-log/$DATASET/${MODEL[i]}/$recipe/test.csv\
      --val_adv attack-log/$DATASET/${MODEL[i]}/$recipe/test.csv\
      --attack_type $recipe\
      --seed $seed --model_params_path $PARAM_PATH\
      --exp_name $EXP_NAME --gpu $GPU --target_model ${TARGET_MODEL[i]}
    done
  done
done
