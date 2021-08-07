RECIPE="bae textfooler pwws tf-adj"

for recipe in $RECIPE
do
  python main.py --test_adv attack-log/imdb/roberta/${recipe}/roberta-base-imdb_${recipe}-test.csv \
   --val_adv attack-log/imdb/roberta/${recipe}/roberta-base-imdb_${recipe}-val.csv \
   --attack_type $recipe
done