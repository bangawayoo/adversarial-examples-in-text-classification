export CUDA_VISIBLE_DEVICES=0,1
MODEL=("bert" "roberta")
DATASET="imdb"
MODEL_DATASET="imdb"
TARGET_MODEL=("textattack/bert-base-uncased-$MODEL_DATASET" "textattack/roberta-base-$MODEL_DATASET")
RECIPE="pwws textfooler bae tf-adj"
EXP_NAME="tmp"
PARAM_PATH="params/reduce_dim_50.json"
SEED_START=0
SEED_END=2
GPU=1

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
