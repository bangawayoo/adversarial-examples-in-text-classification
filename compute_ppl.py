"""
Script computing the ppl of the given texts with GPT-2.
Optionally compute the TPR at given FPR.

Input to the script is *.csv files.
"""

import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from torch.nn.utils import rnn

def process_text(filename, use_original=False):
  def clean_text(t):
    t = t.replace("[", "")
    t = t.replace("]", "")
    return t

  df = pd.read_csv(filename)
  df.loc[df.result_type=='Failed', 'result_type'] = 0
  df.loc[df.result_type=='Successful', 'result_type'] = 1
  attack_success = df.loc[df.result_type==1][['perturbed_text', 'result_type']]
  attack_success = attack_success.rename(columns={'perturbed_text': 'text'})
  if use_original:
    attack_failed = df[['original_text', 'result_type']]
    attack_failed.loc[:, 'result_type'] = 0
  else:
    attack_failed = df.loc[df.result_type == 1][['original_text', 'result_type']]
    attack_failed['result_type'] = 0

  attack_failed = attack_failed.rename(columns={'original_text': 'text'})
  df = pd.concat([attack_failed, attack_success], axis=0)

  if 'nli' in filename: #For NLI dataset, only get the hypothesis, which is attacked
    df['text'] = df['text'].apply(lambda x : x.split('>>>>')[1])
  df['text'] = df['text'].apply(clean_text)

  return df

def compute_loss(model, tokenizer, texts, device):
  encodings = tokenizer.batch_encode_plus(texts, add_special_tokens=True, truncation=True)

  batch_size = 1
  num_batch = len(texts) // batch_size
  eval_loss = 0
  llhs = []

  with torch.no_grad():
    for i in range(num_batch):
      start_idx = i * batch_size;
      end_idx = (i + 1) * batch_size
      x = encodings[start_idx:end_idx]
      ids = torch.LongTensor(x[0].ids)
      ids = ids.to(device)
      llh = model(input_ids=ids, labels=ids)[0] # negative log-likelihood
      llhs.append(llh.item())

  return llhs

def compute_results(df, llhs, fpr_thres=0.05, visualize=True):
  target = df['result_type'].tolist()
  df['nll'] = np.array(llhs)
  df['ppl'] = np.exp(llhs)
  output = llhs

  fpr, tpr, thres = roc_curve(target, llhs)
  mask = (fpr > fpr_thres)
  tpr_at_fpr = np.unique(tpr * mask)[1]  # Minimum number larger than 0

  print(f"TPR at FPR={fpr_thres} : {tpr_at_fpr}")
  if visualize:
    ax = sns.boxplot(x='result_type', y='nll', data=df)
    plt.show()

if __name__ == "__main__" :
  FILENAME = 'results/bert-base-uncased-imdb-textattack-word_thres_0.7.csv'
  DEVICE = torch.device('cuda:1')
  MODEL_ID = 'gpt2-large'

  print(f"Initializing {MODEL_ID}")
  model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(DEVICE)
  model.eval()
  tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID)
  print("Done")

  df = process_text(FILENAME, use_original=False)
  texts = df['text'].tolist()
  print(df.result_type.value_counts())
  llhs = compute_loss(model, tokenizer, texts, DEVICE)
  compute_results(df, llhs, fpr_thres=0.093)






