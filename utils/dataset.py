# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle

from datasets import load_dataset
import numpy as np
import pandas as pd

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

def read_testset_from_csv(filename, use_original=False, split_type='disjoint_subset', split_ratio=0.75, seed=2):
  def clean_text(t):
    t = t.replace("[", "")
    t = t.replace("]", "")
    return t

  filename = args.adv_from_file
  df = pd.read_csv(filename)
  df.loc[df.result_type == 'Failed', 'result_type'] = 0
  df.loc[df.result_type == 'Successful', 'result_type'] = 1
  df.loc[df.result_type == 'Skipped', 'result_type'] = -1

  assert split_type in ['fgws', 'random_sample', 'control_sample', 'control_success', 'attack_scenario'], "Check split type"
  if split_type=='random_sample':
    num_samples = df.shape[0]
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

def read_testset_from_pkl(filename):
  #Input : pkl file name
    # adv_examples.pkl : list of dict with keys below
    # dict_keys(['clean', 'perturbed', 'clean_pred', 'label', 'perturbed_pred', 'perturbed_idxs'])
    # 'clean' and 'perturbed' are list of tokens
  #Output: return pd.DataFrame type similar to read_testset_from_csv()
    # columns: text, result_type, ground_truth_output,
  filename = "adv_examples.pkl"
  with open(filename, 'rb') as h :
    pkl_samples = pickle.load(h)

  df = pd.DataFrame.from_records(pkl_samples)
  adv_index = (df['label']!=df['perturbed_pred']) & (df['label']==df['clean_pred'])
  adv_samples = df.loc[adv_index, ['perturbed', 'label']]
  adv_samples = adv_samples.rename(columns={'perturbed':'text'})
  adv_samples['result_type'] = 1
  clean_samples = df[['clean', 'label']]
  clean_samples = clean_samples.rename(columns={'clean':'text'})
  clean_samples['result_type'] = 0

  testset = pd.concat([adv_samples, clean_samples], axis=0)
  testset = testset.rename(columns={'label':'ground_truth_output'})
  testset['text'] = testset['text'].apply(lambda x: " ".join(x))
  # testset['text'] = testset['text'].apply(lambda x: x.replace(" .", "."))

  return testset