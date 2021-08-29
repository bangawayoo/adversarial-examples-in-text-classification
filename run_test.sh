export CUDA_VISIBLE_DEVICES=0,1
DATASET="imdb"
MODEL="bert"
TARGET_MODEL="textattack/bert-base-uncased-imdb"
RECIPE="pwws textfooler bae tf-adj"
EXP_NAME="Paper-50"
PARAM_PATH="params/attention_key-exclude.json"
GPU=1

python utils/dataset.py $DATASET

for recipe in $RECIPE
do
  for seed in $(seq 0 2)
  do
    python main.py --dataset $DATASET\
    --test_adv attack-log/$DATASET/$MODEL/$recipe/test.csv\
    --val_adv attack-log/$DATASET/$MODEL/$recipe/val.csv\
    --attack_type $recipe\
    --seed $seed --model_params_path $PARAM_PATH\
    --exp_name $EXP_NAME --gpu $GPU --target_model $TARGET_MODEL --tune_params --baseline
  done
done
