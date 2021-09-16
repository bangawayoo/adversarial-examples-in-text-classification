import argparse
import json
import pdb

import torch
from textattack import attack_recipes
from sklearn.decomposition import KernelPCA, PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import random_projection

parser = argparse.ArgumentParser(description="Detect and defense attacked samples")

parser.add_argument("--dataset", default="imdb", type=str,
                    choices=["dbpedia14", "ag-news", "imdb", "yelp", "mnli", "sst2"],
                    help="classification dataset to use")
parser.add_argument("--preprocess", default="standard", type=str,
                    choices=["standard", "fgws"])
parser.add_argument("--target_model", default="textattack/bert-base-uncased-imdb", type=str, #textattack/roberta-base-SST-2
                    help="type of model (textattack pretrained model, path to ckpt)")
parser.add_argument("--test_adv", default="attack-log/imdb/bert/textfooler/test.csv", type=str,
                    help="perturbed texts files with extension csv or pkl")
parser.add_argument("--val_adv", default="attack-log/imdb/bert/textfooler/test.csv", type=str,
                    help="perturbed texts files with extension csv or pkl")
parser.add_argument("--attack_type", default='textfooler', type=str,
                    help="attack type for logging")
parser.add_argument("--exp_name", default='tmp', type=str,
                    help="Name for logging")

parser.add_argument("--fpr_threshold", default=0.10)
parser.add_argument("--compute_bootstrap", default=False, action="store_true")
parser.add_argument("--baseline", default=False, action="store_true")
# parser.add_argument("--split_ratio", default=1.0)

parser.add_argument("--k_tune_range", type=str, default="0 200 5",
                    help="Three int values indicating <start k> <end k> <step>")
parser.add_argument("--tune_params", default=False, action="store_true",
                    help="Whether to use the found best_params.pkl if it exists")
parser.add_argument("--model_params_path", type=str, default="params/attention_key-exclude.json",
                    help="path to json file containing params about probability modeling")

parser.add_argument("--gpu", default='0', type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--mnli_option", default="matched", type=str,
                    choices=["matched", "mismatched"],
                    help="use matched or mismatched test set for MNLI")

args, _ = parser.parse_known_args()


from utils.detection import *
from utils.dataset import *
from utils.logger import *
from utils.miscellaneous import *
from models.wrapper import BertWrapper
from Detector import Detector

#List of hyper-parameters
model_type = args.target_model.replace("/","-")
assert args.attack_type in args.test_adv, f"Attack Type Error: Check if {args.test_adv} is based on {args.attack_type} method"
if args.exp_name:
  args.log_path = f"runs/{args.dataset}/{args.exp_name}/{model_type}/{args.attack_type}"
else:
  args.log_path = f"runs/{args.dataset}/{model_type}/{args.attack_type}"

if __name__ == "__main__":
  if not os.path.isdir(args.log_path):
    os.makedirs(args.log_path)
  logger = Logger(args.log_path, args.seed)
  logger.log.info("Args: "+str(args.__dict__))

  with open(args.model_params_path, "r") as r:
    params = json.load(r)
  logger.log.info("Using params...")
  logger.log.info(params)

  model_wrapper = BertWrapper(args, logger)
  model = model_wrapper.model
  tokenizer = model_wrapper.tokenizer
  model.eval()

  trainvalset, _, key = get_dataset(args)
  text_key, testset_key = key
  trainset, _ = split_dataset(trainvalset, split='trainval', split_ratio=1.0)
  train_features = get_train_features(model_wrapper, args, batch_size=256, dataset=trainset, text_key=text_key, layer=params['layer_param']['cls_layer'])

  feats = []
  for cls, x in enumerate(train_features):
      data = torch.cat(x, dim=0)
      cls_vector = torch.tensor(cls).repeat(data.shape[0], 1)
      feats.append(torch.cat([data, cls_vector], dim=1))

  feats = torch.cat(feats, dim=0)

  if False:
    SAMPLE = True
    if SAMPLE:
      torch.manual_seed(0)
      sample_idx = torch.randperm(len(feats))[:8000]
      sampled_feats = feats[sample_idx, :-1].numpy()
      labels = feats[sample_idx, -1].numpy()

    # reducer = LDA()
    # reduced_feat = reducer.fit_transform(sampled_feats, labels)
    # reducer = random_projection.GaussianRandomProjection(eps=0.5)
    # reduced_feat = reducer.fit_transform(sampled_feats)
    reducer = KernelPCA(n_components=50, kernel='rbf', random_state=0)
    reduced_feat = reducer.fit_transform(sampled_feats)
  else:
      reducer = None
      reduced_feat = feats[:, :-1].numpy()
      labels = feats[:, -1].numpy()

  train_stats = get_stats(reduced_feat, labels, use_shared_cov=False)

  k_s = list(map(lambda x: float(x), args.k_tune_range.split()))
  tune_params = {'topk': {'start':k_s[0], 'end':k_s[1], 'step':k_s[2]}}
  detector = Detector(tune_params, model_wrapper, args.val_adv, train_stats, logger, params, reducer, dataset=args.dataset , seed=args.seed)
  if args.baseline:
    pass
    # detector.test_baseline_PPL(args.test_adv, args.fpr_threshold)
    # detector.test_comb(args.test_adv, args.fpr_threshold)
    # detector.test_baseline(args.test_adv, args.fpr_threshold)
  else:
    if args.tune_params:
      detector.grid_search(fpr_thres=args.fpr_threshold, tune_params=args.tune_params)
    roc, auc, tpr_at_fpr, naive_tpr, conf, testset = detector.test(args.test_adv, args.fpr_threshold)

  if args.compute_bootstrap:
    # Compute bootstrap scores
    target = testset.result_type.values
    scores = compute_bootstrap_score(conf, target, roc, args.fpr_threshold)

    logger.log.info("-----Bootstrapped Results-----")
    for k, v in scores.items():
      logger.log.info(f"{k}: {v:.4f}")


