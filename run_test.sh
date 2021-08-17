DATASET="imdb"
MODEL="roberta"
RECIPE="pwws textfooler tf-adj bae"
EXP_NAME="attention-key-exclude"

python utils/dataset.py $DATASET

for recipe in $RECIPE
do
  for seed in $(seq 1 2)
  do
    python main.py --dataset $DATASET\
    --test_adv attack-log/$DATASET/$MODEL/$recipe/test.csv\
    --val_adv attack-log/$DATASET/$MODEL/$recipe/val.csv\
    --attack_type $recipe\
    --seed $seed\
    --exp_name $EXP_NAME
  done
done

