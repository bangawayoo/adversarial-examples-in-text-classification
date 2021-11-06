export CUDA_VISIBLE_DEVICES=0,1
MODEL=("bert" "roberta")

DATASET="imdb"   #Options: ("imdb" , "ag-news", "sst2")
MODEL_DATASET="imdb" # Change to "SST-2" for "sst2" only
TARGET_MODEL=("textattack/bert-base-uncased-$MODEL_DATASET" "textattack/roberta-base-$MODEL_DATASET")

RECIPE="textfooler pwws bae tf-adj" #Four attack options (No tf-adj for sst2 dataset)
EXP_NAME="tmp"
PARAM_PATH="params/reduce_dim_100.json"
SCEN="s1"  #Options: ("s1" "s2")
ESTIM="MCD"  #Options : ("None", "MCD")

START_SEED=0
END_SEED=0
GPU=0


for ((i=0; i< ${#MODEL[@]}; i++ ));
do
  for recipe in $RECIPE
  do
    python main.py --dataset $DATASET --model_type ${MODEL[i]}\
    --attack_type $recipe --scenario $SCEN --cov_estimator $ESTIM\
    --start_seed $START_SEED --end_seed $END_SEED --model_params_path $PARAM_PATH\
    --exp_name $EXP_NAME --gpu $GPU --target_model ${TARGET_MODEL[i]}

  done
done
