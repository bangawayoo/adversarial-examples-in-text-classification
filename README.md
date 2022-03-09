# Detection of Adversarial Examples in NLP: Benchmark and Baseline via Robust Density Estimation

Official code for *Detection of Adversarial Examples in Text Classification: Benchmark and Baseline via Robust Density Estimation, 
(ACL Findings 2022)*.

## Main Libraries
* python (3.7.0)
* pytorch (1.8.1)
* transformers (4.4.2)
* textattack (0.2.15)

We recommend you to use conda for building the environment.
```
conda env update -n my_env --file environment.yaml
```
If you only wish to use the benchmark dataset without reproducing the experiments in the paper, textattack is not necessary.
Install the dependenceis as needed. 

## Dataset 
Download the generated attacks from [this link](https://drive.google.com/file/d/1eNJn7rX2OE9UKrrknmvClt3IK2cUTRjB/view?usp=sharing) and unzip it. The zip file contains original attacks and adaptive attacks 
described in the paper. 
The directory should look like 
```
./attack-log
    original/
        imdb/
        ag-news/
        sst2/
        yelp/
    stronger/
    strongest/ 
./README.md 
./run_test.sh
...
```


## Applying Your Detection Method 
To follow the experimental settings, we guide you to `AttackLoader.py`.
```
# Initialize Loader 
# You need to specify which scenario, model, attack type, and dataset you are using in args. 
loader = AttackLoader(args, logger)

# Split test and validation set
# Cache will be saved in attack-log/cache/~ 
loader.split_csv_to_testval() 

# Return subsampled testset according to chosen scenario 
sampled, _ = loader.get_attack_from_csv(dtype='test')

# Apply your detection method below 
```


## Reproducing Numbers
The source code relies heavily on TextAttack and transformers. Make sure they are running properly.  
Edit options on `run_test.sh`.
The script will loop through the variables `MODEL, TARGET_MODEL, RECIPE, START_SEED, END_SEED`.  
Some dummy variables exist for purposes of convenience such as `MODEL & TARGET_MODEL` and `DATASET & MODEL_DATASET`.   
Below we provide some description.
```
MODEL=("bert" "roberta") #generic name for models; Options: ("bert", "roberta") 

DATASET="imdb"   #Options: ("imdb" , "ag-news", "sst2")
MODEL_DATASET="imdb" #Change to "SST-2" for "sst2" only
TARGET_MODEL=("textattack/bert-base-uncased-$MODEL_DATASET" "textattack/roberta-base-$MODEL_DATASET")

RECIPE="textfooler pwws bae tf-adj" #Four attack options (No tf-adj for sst2 dataset)
EXP_NAME="tmp" #name for experiment
PARAM_PATH="params/reduce_dim_100.json" #Indicate model parameters (e.g. No PCA, linear PCA, MLE) 
SCEN="s1"  #Scenario (see paper for details); Options: ("s1" "s2") 
ESTIM="MCD"  #Options : ("None", "MCD")

START_SEED=0
END_SEED=0
GPU=0
```
The following command will run the script and log on `./run/$DATASET/$EXP_NAME/$TARGET_MODEL/$recipe`
```
bash run_test.sh 
```

We provide some parameter files used in paper. This can be run by changing `PARAM_PATH`. 
```
./params/reduce_dim_100.json #P=100, kernel: RBF (denoted as RDE) 
./params/reduce_dim_false.json #full dimensions (denoted as MLE)
./params/reduce_dim_100_linear.json #P=100, kernel: linear (standard PCA)
```

Optionally, you can also run the python script.
Checkout the arguments in the script (e.g. baseline, MCD_h)
```
python main.py 
```
