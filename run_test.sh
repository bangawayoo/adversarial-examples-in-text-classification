RECIPE="textfooler tf-adj bae"

for seed in $(seq 0 2)
do
  for recipe in $RECIPE
  do
    python main.py --dataset imdb\
    --test_adv attack-log/imdb/roberta/$recipe/test.csv\
    --val_dav attack-log/imdb/roberta/$recipe/val.csv\
    --attack_type $recipe\
    --seed $seed
  done
done

