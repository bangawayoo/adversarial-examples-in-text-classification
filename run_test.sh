RECIPE="random prioritized genetic pwws"

for recipe in $RECIPE
do
  python main.py --test_adv attack-from-fgws/imdb/${recipe}/${recipe}-test.pkl \
   --val_adv attack-from-fgws/imdb/${recipe}/${recipe}-val.pkl \
   --attack_type $recipe
done
