export CUDA_VISIBLE_DEVICES=0,1
DATASET="imdb"
MODEL=("bert" "roberta")
TARGET_MODEL=("textattack/bert-base-uncased-$DATASET" "textattack/roberta-base-$DATASET")
RECIPE="pwws textfooler bae tf-adj"
EXP_NAME="euclidean"
PARAM_PATH="params/cosine_sim-include.json"
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
      python main_baseline.py --dataset $DATASET\
      --test_adv attack-log/$DATASET/${MODEL[i]}/$recipe/test.csv\
      --val_adv attack-log/$DATASET/${MODEL[i]}/$recipe/test.csv\
      --attack_type $recipe\
      --seed $seed --model_params_path $PARAM_PATH\
      --exp_name $EXP_NAME --gpu $GPU --target_model ${TARGET_MODEL[i]}
    done
  done
done
