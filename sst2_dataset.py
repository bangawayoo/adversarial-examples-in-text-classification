import pandas as pd
from textattack.datasets import Dataset

path = "data/test.csv"
dataset = pd.read_csv(path, header=None, sep='\t')
dataset = list(dataset.itertuples(index=False, name=None))
dataset = Dataset(dataset)
