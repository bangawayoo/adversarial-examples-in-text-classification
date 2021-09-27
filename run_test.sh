export CUDA_VISIBLE_DEVICES=0,1
MODEL=("bert" "roberta")
DATASET="sst2"
MODEL_DATASET="SST-2"
TARGET_MODEL=("textattack/bert-base-uncased-$MODEL_DATASET" "textattack/roberta-base-$MODEL_DATASET")
RECIPE="pwws textfooler bae"
EXP_NAME="rs_all/reduce_dim_2_all"
PARAM_PATH="params/reduce_dim_2.json"
START_SEED=0
END_SEED=2
GPU=1


for ((i=0; i< ${#MODEL[@]}; i++ ));
do
  for recipe in $RECIPE
  do
    python main.py --dataset $DATASET --model_type ${MODEL[i]}\
    --attack_type $recipe\
    --start_seed $START_SEED --end_seed $END_SEED --model_params_path $PARAM_PATH\
    --exp_name $EXP_NAME --gpu $GPU --target_model ${TARGET_MODEL[i]}
  done
done
