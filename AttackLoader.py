import os
import glob
import pdb
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
"""
Loader class called from main.py
 - loads relevant csv file (attack,model,dataset)
 - split into val, test csv and save it to cache
 - random sample necessary file and split into sampled-test-seed{seed} , and save it to cache
"""

class AttackLoader():
  def __init__(self, args, logger, data_type="csv"):
    self.cache_dir = "attack-log/cache"
    if not os.path.isdir(self.cache_dir):
      os.mkdir(self.cache_dir)

    self.logger = logger
    self.scenario = args.scenario
    self.max_adv_num_dict = {'imdb':2000, 'ag-news':2000, 'sst2':1000}
    self.max_adv_num = self.max_adv_num_dict[args.dataset]
    self.args = args

    if data_type == "standard":
      self.root = "attack-log/"
      self.data_dir = os.path.join(self.root, args.dataset)
      self.model_dir = os.path.join(self.data_dir, args.model_type)
      self.csv_dir = os.path.join(self.model_dir, args.attack_type)
      csv_files = glob.glob(os.path.join(self.csv_dir,"*.csv"))
      assert len(csv_files) == 1, f"{len(csv_files)} exists in {self.csv_dir}"
      self.csv_file = csv_files[0]
      self.seed = logger.seed
      self.val_ratio = 0.3 if args.dataset!="sst2" else 0.0
      self.split_csv_to_testval()

  def split_csv_to_testval(self):
    self.logger.log.info(f"Splitting {self.csv_file}")
    np.random.seed(self.seed)
    df = pd.read_csv(self.csv_file)
    num_samples = len(df)
    indices = np.random.permutation(range(num_samples))
    split_point = int(num_samples*self.val_ratio)

    valset = df.iloc[indices[:split_point]]
    if self.args.dataset == "sst2":
      val_path = os.path.join(self.data_dir,"val")
      csv_files = glob.glob(os.path.join(val_path, f"{self.args.model_type}*{self.args.attack_type}.csv"))
      assert len(csv_files) == 1, f"{len(csv_files)} exists in validation path {csv_files}"
      valset = pd.read_csv(csv_files[0])
      val_path = os.path.join(self.cache_dir, "val.csv")
      valset.to_csv(val_path)
    elif self.val_ratio == 0 :
      print(f"Skipping validation set")
    else:
      val_path = os.path.join(self.cache_dir, "val.csv")
      valset.to_csv(val_path)
    testset = df.iloc[indices[split_point:]]
    testpath = os.path.join(self.cache_dir, "test.csv")
    testset.to_csv(testpath)
    self.logger.log.info("test/val split saved in cache")

  def get_attack_from_csv(self, dtype='test', batch_size=64,
                          model_wrapper=None):
    def clean_text(t):
      t = t.replace("[", "")
      t = t.replace("]", "")
      return t

    df = pd.read_csv(os.path.join(self.cache_dir, f"{dtype}.csv"))
    df.loc[df.result_type == 'Failed', 'result_type'] = 0
    df.loc[df.result_type == 'Successful', 'result_type'] = 1
    df.loc[df.result_type == 'Skipped', 'result_type'] = -1

    assert self.scenario in ['fgws', 'random_sample', 'control_sample', 'control_success',
                          's1', 's2'], "Check split type"
    if self.scenario == 's1':
      num_samples = df.shape[0]
      num_adv = (df.result_type == 1).sum()
      """
      Procedure: 
       1. randomly sample N samples from testset and attain adverserial samples.
       2. from the remaining testset randomly sample clean samples (around N)  
      
      How to Choose N: 
      number of random samples to take is determined by (# of desired adv. samples / success rate of adv. attack) 
      (# of desired adv. samples / success rate of adv. attack) = (# of desired adv. samples / # of adv. samples) / (# of total samples) 
      
      split_ratio :  (# of desired adv. samples / # of adv. samples) = max_adv_num / num_adv 
      max_adv_num is dataset dependent and decremented by 10 until attaining this is possible without causing clean/adv class imbalance
      """
      max_adv_num = self.max_adv_num
      adv_sr = num_adv / num_samples
      target_samples = max_adv_num * (1/adv_sr)
      split_ratio = target_samples/num_samples      # ratio to attain max_adv_num number of adv. samples
      while split_ratio >= 0.6 and max_adv_num > 0:
        split_ratio = max_adv_num / num_adv
        max_adv_num -= 10
      if split_ratio >= 0.6 or max_adv_num < 0:
        raise Exception(
          f"Dataset is too small to sample enough adverserial samples. Total: {num_samples}, Adv.: {num_adv}")

      np.random.seed(self.seed)

      # if 'test.csv' in self.csv_file: dtype = 'test'
      # elif 'val.csv' in self.csv_file: dtype = 'val'
      # else: dtype = ''

      rand_idx = np.arange(num_samples)
      np.random.shuffle(rand_idx)

      split_point = int(num_samples * split_ratio)
      split_idx = rand_idx[:split_point]
      split = df.iloc[rand_idx[split_idx]]
      adv = split.loc[split.result_type == 1]
      adv = adv.rename(columns={"perturbed_text": "text"})
      num_adv_samples = adv.shape[0]

      other_split_idx = rand_idx[split_point:split_point + num_adv_samples]
      other_split = df.iloc[other_split_idx].copy()
      clean = other_split  # Use correct and incorrect samples
      clean.loc[:, 'result_type'] = 0
      clean = clean.rename(columns={"original_text": "text"})
      testset = pd.concat([adv, clean], axis=0)
      testset.to_csv(os.path.join(self.cache_dir, f'sampled-{dtype}-{self.seed}.csv'))

    elif self.scenario in ['control_success', 'attack_scenario']:
      assert "not implemented"
      attack_success = df.loc[df.result_type == 1][['perturbed_text', 'result_type', 'ground_truth_output']]
      attack_success = attack_success.rename(columns={'perturbed_text': 'text'})
      use_original = False
      if use_original:
        attack_failed = df[['original_text', 'result_type']]
        attack_failed.loc[:, 'result_type'] = 0
      else:
        text_type = 'perturbed_text' if self.scenario == 'attack_scenario' else 'original_text'
        attack_failed = df.loc[df.result_type == 0][[text_type, 'result_type', 'ground_truth_output']]
      attack_failed = attack_failed.rename(columns={text_type: 'text'})
      testset = pd.concat([attack_failed, attack_success], axis=0)

    elif self.scenario == 'fgws':
      adv_samples = df.loc[df.result_type == 1][['perturbed_text', 'result_type', 'ground_truth_output']]
      adv_samples['result_type'] = 1
      # max_adv_num = min(max_adv_num, len(adv_samples))
      # adv_samples = adv_samples.iloc[:max_adv_num]
      adv_samples = adv_samples.rename(columns={'perturbed_text': 'text'})
      # clean_samples = df.loc[df.result_type != -1][['original_text', 'result_type', 'ground_truth_output']]
      clean_samples = df[
        ['original_text', 'result_type', 'ground_truth_output']]  # Take all samples (correct and incorrect)
      clean_samples['result_type'] = 0
      clean_samples = clean_samples.rename(columns={'original_text': 'text'})
      testset = pd.concat([clean_samples, adv_samples], axis=0)

    elif self.scenario == 's2':
      adv_samples = df.loc[df.result_type == 1][['perturbed_text', 'result_type', 'ground_truth_output']]
      adv_samples['result_type'] = 1
      max_adv_num = min(self.max_adv_num, len(adv_samples))
      adv_samples = adv_samples.iloc[:max_adv_num]
      adv_samples = adv_samples.rename(columns={'perturbed_text': 'text'})
      clean_samples = df.loc[df.result_type ==1][['original_text', 'result_type', 'ground_truth_output']]  # Take all samples (correct and incorrect)
      clean_samples = clean_samples.iloc[:max_adv_num]
      clean_samples['result_type'] = 0
      clean_samples = clean_samples.rename(columns={'original_text': 'text'})
      testset = pd.concat([clean_samples, adv_samples], axis=0)

    if 'nli' in self.csv_file:  # For NLI dataset, only get the hypothesis, which is attacked
      df['original_text'] = df['original_text'].apply(lambda x: x.split('>>>>')[1])
      testset['text'] = testset['text'].apply(lambda x: x.split('>>>>')[1])
    df['original_text'] = df['original_text'].apply(clean_text)
    df['perturbed_text'] = df['perturbed_text'].apply(clean_text)
    testset['text'] = testset['text'].apply(clean_text)

    if model_wrapper:
      self.__sanity_check(df, model_wrapper, batch_size)

    return testset, df

  def get_attack_from_pkl(self, filename, model_wrapper, batch_size):
    with open(filename, 'rb') as h:
      pkl_samples = pickle.load(h)

    ori_len = len(pkl_samples)
    pkl_samples = [i for i in pkl_samples if i is not None]
    reduced_len = len(pkl_samples)
    if ori_len > reduced_len:
      self.logger.log.debug(f"{ori_len - reduced_len} samples removed while attacking.")

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
    self.logger.log.info("Sanity Check for testset")
    target_adv_indices = np.concatenate(target_adv_indices, axis=0)
    adv_pred = np.concatenate(adv_pred, axis=0)
    clean_pred = np.concatenate(clean_pred, axis=0)
    fgws_adv_pred = df['perturbed_pred'].values
    fgws_adv_pred[np.isnan(fgws_adv_pred)] = adv_pred[np.isnan(fgws_adv_pred)]
    adv_pred_diff = (np.not_equal(adv_pred, fgws_adv_pred)).sum()
    clean_pred_diff = (np.not_equal(clean_pred, df['clean_pred'].values)).sum()
    incorrect_indices = np.not_equal(df['clean_pred'].values, df['label'].values)
    self.logger.log.info(f"# of adv. predictions different : {adv_pred_diff}")
    self.logger.log.info(f"# of clean predictions different : {clean_pred_diff}")
    self.logger.log.info(f"Clean Accuracy {correct / total}")
    self.logger.log.info(f"Robust Accuracy {adv_correct / total}")
    self.logger.log.info(f"Adv. Success Rate {target_adv_indices.sum() / total}")
    # Collect adversarial and clean samples
    adv_samples = df[target_adv_indices][['perturbed', 'label']]
    adv_samples = adv_samples.rename(columns={'perturbed': 'text'})
    adv_samples['result_type'] = 1
    clean_samples = df[['clean', 'label']]
    clean_samples = clean_samples.rename(columns={'clean': 'text'})
    clean_samples['result_type'] = 0
    testset = pd.concat([adv_samples, clean_samples], axis=0)
    testset = testset.rename(columns={'label': 'ground_truth_output'})

    return testset, df

  def split_all_csv_to_testval(self, dir_name, val_ratio=None, seed=0):
    """
    splits all raw attack logs to "test/val.csv" in a given dataset directory
    -{dataset}
      -{model}
        -{attack type}
    """
    csv_dir = []
    csv_files = []

    for root, d_names, f_names in os.walk(dir_name):
      flag = False
      for file in f_names:
        if file.endswith(".csv"):
          flag = True
        if "test" in file or "val" in file:
          flag = False
          break
      if flag:
        csv_dir.append(root)

    for dir_ in csv_dir:
      dir_ = os.path.join(dir_, "*.csv")
      files = glob.glob(dir_)
      csv_files.extend(files)

    print(f"Splitting {len(csv_files)} files in {dir_name}:")

    for file in csv_files:
      np.random.seed(seed)
      df = pd.read_csv(file)
      num_samples = len(df)
      indices = np.random.permutation(range(num_samples))
      split_point = int(num_samples * val_ratio)

      dir = os.path.dirname(file)
      csv_name = os.path.basename(file)[:-4]
      valset = df.iloc[indices[:split_point]]
      if val_ratio == 0:
        print(f"Skipping validation set for {file}")
      else:
        val_path = os.path.join(dir, "val.csv")
        valset.to_csv(val_path)
      testset = df.iloc[indices[split_point:]]
      testpath = os.path.join(dir, "test.csv")
      testset.to_csv(testpath)

  def __sanity_check(self, df, model_wrapper, batch_size):
    dataset = df[['perturbed_text', 'original_text']]
    gt = df['ground_truth_output'].tolist()
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
        adv_examples = dataset['perturbed_text'][lower:upper].tolist()
        clean_examples = dataset['original_text'][lower:upper].tolist()
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
    self.logger.log.info("Sanity Check for testset")
    target_adv_indices = np.concatenate(target_adv_indices, axis=0)
    adv_pred = np.concatenate(adv_pred, axis=0)
    clean_pred = np.concatenate(clean_pred, axis=0)
    fgws_adv_pred = df['perturbed_output'].values
    fgws_adv_pred[np.isnan(fgws_adv_pred)] = adv_pred[np.isnan(fgws_adv_pred)]
    adv_pred_diff = (np.not_equal(adv_pred, fgws_adv_pred)).sum()
    clean_pred_diff = (np.not_equal(clean_pred, df['original_output'].values)).sum()
    incorrect_indices = np.not_equal(df['original_output'].values, df['ground_truth_output'].values)
    self.logger.log.info(f"# of adv. predictions different : {adv_pred_diff}")
    self.logger.log.info(f"# of clean predictions different : {clean_pred_diff}")
    self.logger.log.info(f"Clean Accuracy {correct / total}")
    self.logger.log.info(f"Robust Accuracy {adv_correct / total}")
    self.logger.log.info(f"Adv. Success Rate {target_adv_indices.sum() / total}")