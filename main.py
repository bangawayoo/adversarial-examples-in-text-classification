import argparse
import pdb

import numpy as np
from textattack import attack_results
import pandas as pd

parser = argparse.ArgumentParser(description="Detect and defense attacked samples")

parser.add_argument("--dataset", default="imdb", type=str,
                    choices=["dbpedia14", "ag_news", "imdb", "yelp", "mnli", "sst2"],
                    help="classification dataset to use")
parser.add_argument("--preprocess", default="standard", type=str,
                    choices=["standard", "fgws"])
parser.add_argument("--target_model", default="textattack/roberta-base-imdb", type=str, #textattack/roberta-base-SST-2
                    help="type of model (textattack pretrained model, path to ckpt)")
parser.add_argument("--test_adv", default="attack-log/imdb/roberta/bae/test.csv", type=str,
                    help="perturbed texts files with extension csv or pkl")
parser.add_argument("--val_adv", default="attack-log/imdb/roberta/bae/val.csv", type=str,
                    help="perturbed texts files with extension csv or pkl")
parser.add_argument("--attack_type", default='bae', type=str,
                    help="attack type for logging")

parser.add_argument("--fpr_threshold", default=0.093)
parser.add_argument("--compute_bootstrap", default=False, action="store_true")
# parser.add_argument("--split_ratio", default=1.0)

parser.add_argument("--k_tune_range", nargs="+", default="0 255 5",
                    help="Three int values meaning <start k> <end k> <step>")
parser.add_argument("--use_params", default=False, action="store_true",
                    help="Whether to use the found best_params.pkl if it exists")

parser.add_argument("--gpu", default='1', type=str)
parser.add_argument("--seed", default=2, type=int)
parser.add_argument("--mnli_option", default="matched", type=str,
                    choices=["matched", "mismatched"],
                    help="use matched or mismatched test set for MNLI")

args, _ = parser.parse_known_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from utils.detection import *
from utils.dataset import *
from utils.logger import *
from utils.miscellaneous import *
from models.wrapper import BertWrapper
from Detector import Detector

#List of hyper-parameters
LAYER = -1
model_type = args.target_model.replace("/","-")
assert args.attack_type in args.test_adv, f"Attack Type Error: Check if {args.test_adv} is based on {args.attack_type} method"
args.log_path = f"runs/{args.dataset}/{model_type}/{args.attack_type}"

if __name__ == "__main__":
  if not os.path.isdir(args.log_path):
    os.makedirs(args.log_path)
  logger = Logger(args.log_path, args.seed)
  logger.log.info("Args: "+str(args.__dict__))

  model_wrapper = BertWrapper(args, logger)
  model = model_wrapper.model
  tokenizer = model_wrapper.tokenizer
  model.eval()

  trainvalset, _, key = get_dataset(args)
  text_key, testset_key = key
  trainset, _ = split_dataset(trainvalset, split='trainval', split_ratio=1.0)
  train_features = get_train_features(model_wrapper, args, batch_size=512, dataset=trainset, text_key=text_key, layer=LAYER)
  train_stats = get_stats(train_features, use_shared_cov=True)

  k_s = list(map(lambda x: int(x), args.k_tune_range.split()))
  tune_params = {'topk': {'start':k_s[0], 'end':k_s[1], 'step':k_s[2]}}
  params = {
    "model_param": {'type': 'cosine_sim', 'normalization': 'softmax', 'temperature': 1.0,
                    'attention_type': 'query', 'exclude_cls_token':True},
    'layer_param': {'cls_layer': -1, 'num_layer': 1},
    'prob_param': {'choose_type': 'topk', 'topk': 0, 'sum_heads': True, 'p': 0}}
  detector = Detector(tune_params, model_wrapper, args.val_adv, train_stats, logger, params, seed=args.seed)
  detector.grid_search(fpr_thres=args.fpr_threshold, use_existing_params=args.use_params)
  roc, auc, tpr_at_fpr, conf, testset = detector.test(args.test_adv, args.fpr_threshold)
  logger.save_metric()

  if args.compute_bootstrap:
    # Compute bootstrap scores
    target = testset.result_type.values
    scores = compute_bootstrap_score(conf, target, roc, args.fpr_threshold)

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

