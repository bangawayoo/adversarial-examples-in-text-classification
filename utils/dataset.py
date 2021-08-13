# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import re

import torch
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.preprocess import *


def load_data(args):
    if args.dataset == "ag_news":
        dataset = load_dataset("ag_news")
        num_labels = 4
    elif args.dataset == "imdb":
        dataset = load_dataset("imdb", ignore_verifications=True)
        num_labels = 2
    elif args.dataset == "yelp":
        dataset = load_dataset("yelp_polarity")
        num_labels = 2
    elif args.dataset == "mnli":
        dataset = load_dataset("glue", "mnli")
        num_labels = 3
    elif args.dataset == "sst2":
        dataset = load_dataset("glue", "sst2")
        num_labels = 2
    dataset = dataset.shuffle(seed=0)
    
    return dataset, num_labels


def get_dataset(args):
  # Get train data and split 20% with val.
  dataset, num_labels = load_data(args)
  if args.dataset == 'mnli':
    text_key = None
    testset_key = 'validation_%s' % args.mnli_option
  else:
    text_key = 'text' if (args.dataset in ["ag_news", "imdb", "yelp"]) else 'sentence'
    testset_key = 'test' if (args.dataset in ["ag_news", "imdb", "yelp"]) else 'validation'

  split = 'train'
  trainvalset = dataset[split]
  testset = dataset[testset_key]
  return trainvalset, testset, (text_key, testset_key)


def split_dataset(dataset, split='trainval', split_ratio=0.8):
  num_samples = len(dataset)
  if split == 'trainval':
    indices = np.random.permutation(range(num_samples))
    train_idx, val_idx = indices[:int(num_samples * split_ratio)], indices[int(num_samples * split_ratio):]
    trainset, valset = dataset[train_idx], dataset[val_idx]
    return trainset, valset
  else:
    testset = dataset[range(num_samples)]
    return testset

def read_testset_from_csv(filename, use_original=False, split_type='random_sample', seed=0):
  def clean_text(t):
    t = t.replace("[", "")
    t = t.replace("]", "")
    return t

  # filename = args.adv_from_file
  df = pd.read_csv(filename)
  df.loc[df.result_type == 'Failed', 'result_type'] = 0
  df.loc[df.result_type == 'Successful', 'result_type'] = 1
  df.loc[df.result_type == 'Skipped', 'result_type'] = -1

  assert split_type in ['fgws', 'random_sample', 'control_sample', 'control_success', 'attack_scenario'], "Check split type"
  if split_type=='random_sample':
    num_samples = df.shape[0]
    num_adv = (df.result_type==1).sum()

    for num_sample in range(500,0,-100):
      split_ratio = (num_sample/num_adv) # Ensure minimum of 500 adversarial samples are used if possible
      if split_ratio < 1 :
        break
    if split_ratio >= 1:
      raise Exception(f"Too small data for sampling enough adverserial samples. Total: {num_sample}, Adv.: {num_adv}")

    np.random.seed(seed)
    rand_idx =  np.arange(num_samples)
    np.random.shuffle(rand_idx)

    split_point = int(num_samples*split_ratio)
    split_idx = rand_idx[:split_point]
    split = df.iloc[rand_idx[split_idx]]
    adv = split.loc[split.result_type==1]
    adv = adv.rename(columns={"perturbed_text":"text"})
    num_adv_samples = adv.shape[0]

    other_split_idx = rand_idx[split_point:split_point+num_adv_samples]
    other_split = df.iloc[other_split_idx]
    clean = other_split # Use correct and incorrect samples
    clean['result_type'] = 0
    clean = clean.rename(columns={"original_text": "text"})
    testset = pd.concat([adv, clean], axis=0)

    if 'test.csv' in filename:
      dtype = 'test'
    elif 'val.csv' in filename:
      dtype = 'val'
    else:
      dtype = ''

    file_dir = os.path.dirname(filename)
    file_dir = os.path.join(file_dir, 'random_sample')
    if not os.path.isdir(file_dir):
      os.mkdir(file_dir)
    testset.to_csv(os.path.join(file_dir, f'{dtype}-{seed}.csv'))

  elif split_type in ['control_success', 'attack_scenario']:
    attack_success = df.loc[df.result_type == 1][['perturbed_text', 'result_type', 'ground_truth_output']]
    attack_success = attack_success.rename(columns={'perturbed_text': 'text'})
    if use_original:
      attack_failed = df[['original_text', 'result_type']]
      attack_failed.loc[:, 'result_type'] = 0
    else:
      text_type = 'perturbed_text' if split_type == 'attack_scenario' else 'original_text'
      attack_failed = df.loc[df.result_type==0][[text_type, 'result_type', 'ground_truth_output']]
    attack_failed = attack_failed.rename(columns={text_type: 'text'})
    testset = pd.concat([attack_failed, attack_success], axis=0)

  elif split_type=='fgws':
    adv_samples = df.loc[df.result_type == 1][['perturbed_text', 'result_type', 'ground_truth_output']]
    adv_samples['result_type'] = 1
    adv_samples = adv_samples.rename(columns={'perturbed_text': 'text'})
    # clean_samples = df.loc[df.result_type != -1][['original_text', 'result_type', 'ground_truth_output']]
    clean_samples = df[['original_text', 'result_type', 'ground_truth_output']] # Take all samples (correct and incorrect)
    clean_samples['result_type'] = 0
    clean_samples = clean_samples.rename(columns={'original_text': 'text'})
    testset = pd.concat([clean_samples, adv_samples], axis=0)

  if 'nli' in filename:  # For NLI dataset, only get the hypothesis, which is attacked
    df['original_text'] = df['original_text'].apply(lambda x: x.split('>>>>')[1])
    testset['text'] = testset['text'].apply(lambda x: x.split('>>>>')[1])
  df['original_text'] = df['original_text'].apply(clean_text)
  df['perturbed_text'] = df['perturbed_text'].apply(clean_text)
  testset['text'] = testset['text'].apply(clean_text)

  return testset, df


def read_testset_from_pkl(filename, model_wrapper, batch_size=128, logger=None):
  with open(filename, 'rb') as h :
    pkl_samples = pickle.load(h)

  ori_len = len(pkl_samples)
  pkl_samples = [i for i in pkl_samples if i is not None]
  reduced_len = len(pkl_samples)
  if ori_len > reduced_len:
    logger.log.debug(f"{ori_len-reduced_len} samples removed while attacking.")

  df = pd.DataFrame.from_records(pkl_samples)

  dataset = df[['perturbed', 'clean']]
  gt = df['label'].tolist()
  # Compute Acc. on dataset
  num_samples = len(dataset)
  num_batches = int((num_samples // batch_size) + 1)
  target_adv_indices = []

  correct = 0
  adv_correct = 0
  total = 0
  adv_pred = []
  clean_pred = []

  with torch.no_grad():
    for i in tqdm(range(num_batches)):
      lower = i * batch_size
      upper = min((i + 1) * batch_size, num_samples)
      adv_examples = dataset['perturbed'][lower:upper].tolist()
      clean_examples = dataset['clean'][lower:upper].tolist()
      labels = gt[lower:upper]

      y = torch.LongTensor(labels).to(model_wrapper.model.device)
      output = model_wrapper.inference(adv_examples)
      preds = torch.max(output.logits, dim=1).indices
      adv_pred.append(preds.cpu().numpy())
      adv_correct += y.eq(preds).sum().item()
      adv_error_idx = preds.ne(y)

      output = model_wrapper.inference(clean_examples)
      preds = torch.max(output.logits, dim=1).indices
      clean_pred.append(preds.cpu().numpy())
      correct += y.eq(preds).sum().item()
      clean_correct_idx = preds.eq(y)
      total += preds.size(0)

      target_adv_idx = torch.logical_and(adv_error_idx, clean_correct_idx)
      target_adv_indices.append(target_adv_idx.cpu().numpy())

  """
  Sanity Check : prediction results should be equivalent to FGWS predictions 
  """
  logger.log.info("Sanity Check for testset")
  target_adv_indices = np.concatenate(target_adv_indices, axis=0)
  adv_pred = np.concatenate(adv_pred, axis=0)
  clean_pred = np.concatenate(clean_pred, axis=0)
  fgws_adv_pred = df['perturbed_pred'].values
  fgws_adv_pred[np.isnan(fgws_adv_pred)] = adv_pred[np.isnan(fgws_adv_pred)]
  adv_pred_diff = (np.not_equal(adv_pred, fgws_adv_pred)).sum()
  clean_pred_diff = (np.not_equal(clean_pred, df['clean_pred'].values)).sum()
  incorrect_indices = np.not_equal(df['clean_pred'].values, df['label'].values)
  logger.log.info(f"# of adv. predictions different : {adv_pred_diff}")
  logger.log.info(f"# of clean predictions different : {clean_pred_diff}")
  logger.log.info(f"Clean Accuracy {correct/total}")
  logger.log.info(f"Robust Accuracy {adv_correct/total}")
  logger.log.info(f"Adv. Success Rate {target_adv_indices.sum() / total}")

  # Collect adversarial and clean samples
  adv_samples = df[target_adv_indices][['perturbed', 'label']]
  adv_samples = adv_samples.rename(columns={'perturbed':'text'})
  adv_samples['result_type'] = 1
  clean_samples = df[['clean', 'label']]
  clean_samples = clean_samples.rename(columns={'clean':'text'})
  clean_samples['result_type'] = 0

  testset = pd.concat([adv_samples, clean_samples], axis=0)
  testset = testset.rename(columns={'label':'ground_truth_output'})

  return testset

def split_csv_to_testval(dir_name, val_ratio, seed=0):
  """
  Recursively search for *.csv and create csv with test and val set
  """
  csv_files = []
  for root, d_names, f_names in os.walk(dir_name):
    found = [os.path.join(root,i) for i in f_names if (i.endswith(".csv")) and ('test' not in i and 'val' not in i)]
    csv_files.extend(found)

  print(f"Found {len(csv_files)} files:")
  print(csv_files)
  for file in csv_files:
    np.random.seed(seed)
    df = pd.read_csv(file)
    num_samples = len(df)
    indices = np.random.permutation(range(num_samples))
    split_point = int(num_samples*val_ratio)

    dir = os.path.dirname(file)
    csv_name = os.path.basename(file)[:-4]
    valset = df.iloc[indices[:split_point]]
    val_path = os.path.join(dir, "val.csv")
    valset.to_csv(val_path)
    testset = df.iloc[indices[split_point:]]
    testpath = os.path.join(dir, "test.csv")
    testset.to_csv(testpath)


if __name__ == "__main__":
  seed=0
  split_csv_to_testval("attack-log/imdb/", val_ratio=0.3, seed=seed)