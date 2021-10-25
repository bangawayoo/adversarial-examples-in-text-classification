export CUDA_VISIBLE_DEVICES=0,1
MODEL=("bert" "roberta")
MODEL=("bert" "roberta")

DATASET="imdb"   #Options: ("imdb" , "ag-news", "sst2")
MODEL_DATASET="imdb" # Change to "SST-2" for "sst2" only

#TARGET_MODEL=("textattack/bert-base-uncased-$MODEL_DATASET" "textattack/roberta-base-$MODEL_DATASET")
TARGET_MODEL=("textattack/bert-base-uncased-$MODEL_DATASET" "textattack/roberta-base-$MODEL_DATASET")


#RECIPE="pwws textfooler bae tf-adj"  #No tf-adj for "sst2"
RECIPE="textfooler pwws bae tf-adj"
EXP_NAME="s2/MCD"
PARAM_PATH="params/reduce_dim_100.json"
SCEN="s2"
ESTIM="MCD"
START_SEED=0
END_SEED=2
GPU=0


for ((i=0; i< ${#MODEL[@]}; i++ ));
do
  for recipe in $RECIPE
  do
    python main.py --dataset $DATASET --model_type ${MODEL[i]}\
    --attack_type $recipe --scenario $SCEN --cov_estimator $ESTIM\
    --start_seed $START_SEED --end_seed $END_SEED --model_params_path $PARAM_PATH\
    --exp_name $EXP_NAME --gpu $GPU --target_model ${TARGET_MODEL[i]}
    exit
  done
done
exit
#
MODEL=("bert" "roberta")

DATASET="ag-news"   #Options: ("imdb" , "ag-news", "sst2")
MODEL_DATASET="ag-news" # Change to "SST-2" for "sst2" only

#TARGET_MODEL=("textattack/bert-base-uncased-$MODEL_DATASET" "textattack/roberta-base-$MODEL_DATASET")
TARGET_MODEL=("textattack/bert-base-uncased-$MODEL_DATASET" "textattack/roberta-base-$MODEL_DATASET")


#RECIPE="pwws textfooler bae tf-adj"  #No tf-adj for "sst2"
RECIPE="tf-adj bae pwws textfooler"
EXP_NAME="s2/MCD"
PARAM_PATH="params/reduce_dim_100.json"
SCEN="s2"
ESTIM="MCD"

START_SEED=0
END_SEED=2
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


##
MODEL=("bert" "roberta")

DATASET="sst2"   #Options: ("imdb" , "ag-news", "sst2")
MODEL_DATASET="SST-2" # Change to "SST-2" for "sst2" only

#TARGET_MODEL=("textattack/bert-base-uncased-$MODEL_DATASET" "textattack/roberta-base-$MODEL_DATASET")
TARGET_MODEL=("textattack/bert-base-uncased-$MODEL_DATASET" "textattack/roberta-base-$MODEL_DATASET")


#RECIPE="pwws textfooler bae tf-adj"  #No tf-adj for "sst2"
RECIPE="textfooler pwws bae"
EXP_NAME="s2/MCD/"
PARAM_PATH="params/reduce_dim_100.json"
SCEN="s2"
ESTIM="MCD"
START_SEED=0
END_SEED=2
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

