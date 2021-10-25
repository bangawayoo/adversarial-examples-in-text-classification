import argparse
import json
import os.path
import shutil
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch


parser = argparse.ArgumentParser(description="Detect and defense attacked samples")

parser.add_argument("--dataset", default="imdb", type=str,
                    choices=["dbpedia14", "ag-news", "imdb", "yelp", "mnli", "sst2"],
                    help="classification dataset to use")
parser.add_argument("--preprocess", default="standard", type=str,
                    choices=["standard", "fgws"])
parser.add_argument("--data_type", default="standard", type=str,
                    choices=["standard", "fgws"])
parser.add_argument("--target_model", default="textattack/bert-base-uncased-imdb", type=str, #textattack/roberta-base-SST-2
                    help="name of model (textattack pretrained model, path to ckpt)")
parser.add_argument("--model_type", type=str, help="model type (e.g. bert, roberta, cnn)")
parser.add_argument("--scenario", type=str, help="scenario that determines how the configure the adv. dataset")
parser.add_argument("--use_val", default=False, action='store_true')
parser.add_argument("--cov_estimator", type=str, help="covarianc esitmator",
                    choices=["OAS", "MCD", "None"])


parser.add_argument("--pkl_test_path", default=" ", type=str,
                    help="perturbed texts files with extension csv or pkl")
parser.add_argument("--pkl_val_path", default=" ", type=str,
                    help="perturbed texts files with extension csv or pkl")
parser.add_argument("--attack_type", default='textfooler', type=str,
                    help="attack type for logging")
parser.add_argument("--exp_name", default='tmp', type=str,
                    help="Name for logging")

parser.add_argument("--fpr_threshold", default=0.10)
parser.add_argument("--compute_bootstrap", default=False, action="store_true")
parser.add_argument("--baseline", default=False, action="store_true")

parser.add_argument("--tune_params", default=False, action="store_true",
                    help="Whether to use the found best_params.pkl if it exists")
parser.add_argument("--model_params_path", type=str, default="params/attention_key-exclude.json",
                    help="path to json file containing params about probability modeling")

parser.add_argument("--gpu", default='0', type=str)
parser.add_argument("--start_seed", default=0, type=int)
parser.add_argument("--end_seed", default=0, type=int)
parser.add_argument("--mnli_option", default="matched", type=str,
                    choices=["matched", "mismatched"],
                    help="use matched or mismatched test set for MNLI")

args, _ = parser.parse_known_args()

from textattack import attack_recipes

from utils.detection import *
from utils.dataset import *
from utils.logger import *
from utils.miscellaneous import *
from models.wrapper import BertWrapper
from Detector import Detector
from AttackLoader import AttackLoader

model_type = args.target_model.replace("/","-")
# assert args.attack_type in args.test_adv, f"Attack Type Error: Check if {args.test_adv} is based on {args.attack_type} method"
if args.exp_name:
  args.log_path = f"runs/{args.dataset}/{args.exp_name}/{model_type}/{args.attack_type}"
else:
  args.log_path = f"runs/{args.dataset}/{model_type}/{args.attack_type}"

if __name__ == "__main__":
  if not os.path.isdir(args.log_path):
    os.makedirs(args.log_path)
  logger = Logger(args.log_path)
  logger.log.info("Args: "+str(args.__dict__))

  with open(args.model_params_path, "r") as r:
    params = json.load(r)
  num_params = len(glob.glob(os.path.join(args.log_path, "*.json")))
  shutil.copyfile(args.model_params_path, os.path.join(args.log_path, f"params-{num_params}.json"))
  logger.log.info("Using params...")
  logger.log.info(params)

  model_wrapper = BertWrapper(args, logger)
  model = model_wrapper.model
  tokenizer = model_wrapper.tokenizer
  model.eval()

  trainvalset, _, key = get_dataset(args)
  text_key, testset_key = key
  trainset, _ = split_dataset(trainvalset, split='trainval', split_ratio=1.0)
  feats = get_train_features(model_wrapper, args, batch_size=256, dataset=trainset, text_key=text_key, layer=params['layer_param']['cls_layer'])
  feats = feats.numpy()
  reduced_feat, labels, reducer, scaler = preprocess_features(feats, params, args, logger)

  train_stats, estimators = get_stats(reduced_feat, labels, cov_estim_name=args.cov_estimator, use_shared_cov=params['shared_cov'], params=params)
  naive_train_stats, naive_estimators = get_stats(reduced_feat, labels, cov_estim_name="None", use_shared_cov=params['shared_cov'])
  all_train_stats = [naive_train_stats, train_stats]
  all_estimators = [naive_estimators, estimators]

  visualize = False
  if visualize:
    dir_name = os.path.dirname(args.log_path)
    path_to_feat = os.path.join(dir_name, 'feats.txt')
    feat_n_label = np.concatenate([reduced_feat, labels[:,np.newaxis]], axis=-1)
    np.savetxt(path_to_feat, feat_n_label)
    for cls_idx, mu_n_cov in enumerate(train_stats):
      np.save(os.path.join(dir_name, f"cls{cls_idx}-cov.npy"), mu_n_cov[1])

    for name, stat in zip(['naive', 'robust'], all_train_stats):
      for idx, (mu, cov) in enumerate(stat):
        spectrum = np.linalg.eigvals(cov)
        path_to_csv = os.path.join(os.path.dirname(args.log_path), 'spectrum.csv')
        with open(path_to_csv, 'a') as f:
          wr = csv.writer(f)
          wr.writerow([name, max(spectrum), min(spectrum)])
        kappa = max(spectrum) / min(spectrum)
        plt.matshow(cov)
        plt.title(f"Cond.:{kappa:.3e} Max:{max(spectrum):.3e} Min: {min(spectrum):.3e}")
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.savefig(os.path.join(os.path.dirname(args.log_path), f"{name}-cls{idx}.png"))
    exit()

  for s in range(args.start_seed, args.end_seed+1):
    logger.set_seed(s)
    loader = AttackLoader(args, logger, data_type=args.data_type)
    detector = Detector(model_wrapper, all_train_stats, loader, logger, params, (scaler, reducer, all_estimators, args.cov_estimator), use_val=args.use_val , dataset=args.dataset , seed=s)
    if args.baseline:
      detector.test_baseline_PPL(args.fpr_threshold)
      # detector.test_baseline(args.fpr_threshold)
    else:
      roc, auc, tpr_at_fpr, naive_tpr, conf, testset = detector.test(args.fpr_threshold, args.pkl_test_path)

    if args.compute_bootstrap:
      # Compute bootstrap scores
      target = testset.result_type.values
      scores = compute_bootstrap_score(conf, target, roc, args.fpr_threshold)

      logger.log.info("-----Bootstrapped Results-----")
      for k, v in scores.items():
        logger.log.info(f"{k}: {v:.4f}")


