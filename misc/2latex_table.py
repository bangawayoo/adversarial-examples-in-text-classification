import pandas as pd
import os
import glob

##
"""
Writes line for latex table for given path to attacks and csv file
"""
dataset = 'ag-news'
model = f'textattack-roberta-base-{dataset}' if dataset != 'sst2' else 'textattack-roberta-base-SST-2'
# model = f'textattack-bert-base-uncased-{dataset}' if dataset != 'sst2' else 'textattack-bert-base-uncased-SST-2'

names = ['PPL', 'FGWS', 'MLE', 'RDE(-MCD)','RDE']
names = ['naive-mahal', 'naive-mahal']

file_names = ['ppl.csv', 'fgws.csv', 'mle.csv', 'naive-mahal.csv', 'MCD-mahal.csv']
file_names = ['naive-mahal.csv', 'MCD-mahal.csv']

# attack_dirs = ['textfooler', 'pwws', 'bae', 'tf-adj']
attack_dirs = ['textfooler', 'pwws']
if dataset == 'sst2':
  attack_dirs = ['textfooler', 'pwws', 'bae']
exp_path = f'./runs/{dataset}/s1/tune/PCA300/{model}/'


output = ""

for n, fn in zip(names, file_names):

  sample_lim = 3

  output += f"& {n} &"
  metrics = {'tpr' :[], 'auc':[], 'f1':[]}
  for att in attack_dirs:
    csv_path = os.path.join(exp_path, att, fn)
    assert os.path.isfile(csv_path), f"{csv_path} does not exist"
    raw = pd.read_csv(csv_path)
    if sample_lim:
      sample_lim = min(sample_lim, len(raw))
      raw_cut = raw.iloc[:sample_lim]
    print(f"{len(raw)} outputs for {csv_path}")

    mean_series = raw_cut.mean()
    std_series = raw_cut.std() / sample_lim
    tpr, f1, auc = mean_series['tpr'], mean_series["f1"], mean_series['auc']
    tpr_std, f1_std, auc_std = std_series['tpr'], std_series['f1'], std_series['auc']

    # output += f"{tpr*100:.1f}& {f1*100:.1f}& {auc*100:.1f}   "
    if len(raw) == 1:
      output += f"{tpr*100:.1f}& {f1*100:.1f}& {auc*100:.1f} &"
    else:
      output += f"{tpr*100:.1f}$\pm${tpr_std*100:.1f} & {f1*100:.1f}$\pm${f1_std*100:.1f}& {auc*100:.1f}$\pm${auc_std*100:.1f}&"
  output = output[:-1]
  output += " \\\\ \n"


print(output)