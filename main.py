import argparse
import pdb

import numpy as np
from textattack import attack_results
import pandas as pd

parser = argparse.ArgumentParser(description="Detect and defense attacked samples")

parser.add_argument("--dataset", default="sst2", type=str,
                    choices=["dbpedia14", "ag_news", "imdb", "yelp", "mnli", "sst2"],
                    help="classification dataset to use")
parser.add_argument("--preprocess", default="standard", type=str,
                    choices=["standard", "fgws"])
parser.add_argument("--target_model", default="textattack/roberta-base-SST-2", type=str, #textattack/roberta-base-SST-2
                    help="type of model (textattack pretrained model, path to ckpt)")
parser.add_argument("--adv_from_file", default="attack-from-fgws/sst2/random-test.pkl", type=str,
                    help="perturbed texts from csv or pkl")
parser.add_argument("--attack_type", default='random', type=str,
                    help="attack type for logging")
parser.add_argument("--fpr_threshold", default=0.114)
parser.add_argument("--split_ratio", default=1.0)

parser.add_argument("--gpu", default='1', type=str)
parser.add_argument("--mnli_option", default="matched", type=str,
                    choices=["matched", "mismatched"],
                    help="use matched or mismatched test set for MNLI")

args, _ = parser.parse_known_args()

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# from utils.defense import *
from utils.detection import *
from utils.dataset import *
from utils.preprocess import *
from utils.logger import *
from utils.miscellaneous import *
from models.wrapper import BertWrapper

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

#List of hyper-parameters
REFINE_TOP_K=6
ADV_RATIO=None
LAYER = -1

model_type = args.target_model.replace("/","-")
assert args.attack_type in args.adv_from_file, f"Attack Type Error: Check if {args.adv_from_file} is based on {args.attack_type} method"
args.log_path = f"runs/{args.dataset}/{model_type}/{args.attack_type}"

if __name__ == "__main__":
  if not os.path.isdir(args.log_path):
    os.makedirs(args.log_path)
  logger = Logger(args.log_path)

  model_wrapper = BertWrapper(args, logger)
  model = model_wrapper.model
  tokenizer = model_wrapper.tokenizer
  model.eval()

  trainvalset, _, key = get_dataset(args)
  text_key, testset_key = key
  trainset, _ = split_dataset(trainvalset, split='trainval', split_ratio=args.split_ratio)
  train_features = get_train_features(model_wrapper, args, batch_size=512, dataset=trainset, text_key=text_key, layer=LAYER)
  train_stats = get_stats(train_features, use_shared_cov=True)

  if args.adv_from_file.endswith(".csv"):
    testset, raw_testset = read_testset_from_csv(args.adv_from_file, use_original=False, split_type='random_sample', split_ratio=0.6, seed=2)  # returns pandas dataframe
  if args.adv_from_file.endswith(".pkl"):
    testset = read_testset_from_pkl(args.adv_from_file, model_wrapper, batch_size=128, logger=logger)  # returns pandas dataframe
  logger.log.info(f"--------Loaded {args.adv_from_file}...---------")

  total, adv_count = testset.result_type.value_counts().sum(), testset.result_type.value_counts()[1]
  logger.log.info(f"Percentage of Adv. Samples: {adv_count}/{total} : {adv_count / (total)}")

  texts = testset['text'].tolist()
  logger.log.info("------Building test features...-------")

  best = 0
  best_k = 0
  best_roc = None
  start_k, end_k, step_k = 0, 60, 5
  start_p, end_p, step_p = 0, 1, 1
  for k in range(start_k, end_k, step_k):
    for p in np.arange(start_p, end_p, step_p):
      print(k)
      params = {"model_param": {'type': 'attention', 'normalization': 'softmax', 'tempearture': 1.0, 'attention_type':'key'},
                'layer_param': {'cls_layer': -1, 'num_layer':1},
                'prob_param': {'choose_type':'topk', 'topk':k, 'sum_heads':True, 'p':p}}

      test_features, probs = get_test_features(model_wrapper, batch_size=128, dataset=texts, params=params, logger=logger)
      confidence, conf_indices, distance = compute_dist(test_features, train_stats, distance_type="euclidean", use_marginal=False)

      if probs.dim() == 1:
        probs = probs.unsqueeze(-1)
      conf_indices = torch.max(distance + probs, dim=1).indices
      gt = torch.tensor(testset.loc[testset['result_type']==0, 'ground_truth_output'].values)
      correct = conf_indices[(testset.result_type==0).values].eq(gt).sum()
      logger.log.info(f"Accuracy of hard clustering on {correct}/{gt.numel()}: {correct/gt.numel()}")

      num_nans = sum(probs==-float("Inf"))
      if num_nans != 0 :
        logger.log.info(f"Warning : {num_nans} Nans in conditional probability")
        probs[probs==-float("inf")] = -1e6
      refined_confidence = confidence + probs.squeeze()
      refined_confidence[refined_confidence==-float("Inf")] = -1e6
      refined_confidence[torch.isnan(refined_confidence)] = -1e6

      # Detect attacks for correctly classified samples (unnecessary for fgws, random_sample settings)
      fpr_thres = args.fpr_threshold
      adv_count = testset.loc[testset['result_type']==1].shape[0]
      correct_idx = np.array(testset['result_type']!=-1)
      correct_set = testset.loc[correct_idx]

      logger.log.info("-----Results for Baseline OOD------")
      roc, auc, _ = detect_attack(correct_set, confidence[correct_idx], conf_indices[correct_idx], fpr_thres,
                               visualize=True, logger=logger, mode="Baseline")
      logger.log.info("-----Results for Hierarchical OOD------")
      roc, auc, tpr_at_fpr = detect_attack(correct_set, refined_confidence[correct_idx], conf_indices[correct_idx], fpr_thres,
                               visualize=True, logger=logger, mode="Hierarchical")
      logger.log.info("-----Results for Conditional Probability OOD------")
      _, _, _ = detect_attack(correct_set, probs[correct_idx], conf_indices[correct_idx], fpr_thres,
                               visualize=True, logger=logger, mode="conditional")
      if tpr_at_fpr > best:
        best = tpr_at_fpr
        best_k = k
        best_roc = roc
        best_conf = refined_confidence

  logger.log.info(f"Best : {best} at p={best_k}")
  # Compute bootstrap scores
  target = testset.result_type.values
  scores = compute_bootstrap_score(best_conf, target, best_roc, fpr_thres)

  logger.log.info("-----Bootstrapped Results-----")
  for k, v in scores.items():
    logger.log.info(f"{k}: {v:.4f}")

  """
  Inference Stage:
  1. For all test set, predict whether it is adv samples or not (hyper-params : adv. ratio/ adv_count) ; def predict_attack()
  2. For adv. samples, refine then predict 
  3. For clean samples, predict   
  """
  # print("Inference Stage...")
  # predict_attack(testset, confidence, conf_indices, fpr_thres, visualize=False, by_class=False, adv_ratio=ADV_RATIO)
  #
  # detected_adv, y = testset.loc[testset['pred_type']==1, 'text'], testset.loc[testset['pred_type']==1, 'ground_truth_output']
  # detected_adv, y = raw_testset['perturbed_text'], raw_testset['ground_truth_output']
  # adv_cor, adv_total = refine_predict_by_topk(model, tokenizer, REFINE_TOP_K, False, 64, detected_adv.tolist(), y.tolist())
  # # adv_cor, adv_total = refine_predict_by_prob(model, tokenizer, 128, detected_adv.tolist(), y.tolist(), 0.05)
  #
  # detected_clean, y = testset.loc[testset['pred_type']==0, 'text'], testset.loc[testset['pred_type']==0, 'ground_truth_output']
  # detected_clean, y = testset.loc[testset['result_type']==1, 'text'], testset.loc[testset['result_type']==1, 'ground_truth_output']
  # clean_cor, clean_total = predict_clean_samples(model, tokenizer, 128, detected_clean.tolist(), y.tolist())
  # print(f"Total Acc. {(adv_cor + clean_cor)/ (adv_total+clean_total):.3f}")

