import argparse
import json
import pdb
import torch


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

from textattack import attack_recipes
from sklearn.decomposition import KernelPCA, PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import random_projection

from utils.detection import *
from utils.dataset import *
from utils.logger import *
from utils.miscellaneous import *
from models.wrapper import BertWrapper
from Detector import Detector

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
  feats = get_train_features(model_wrapper, args, batch_size=256, dataset=trainset, text_key=text_key, layer=params['layer_param']['cls_layer'])

  if params['reduce_dim']['do']:
    if params['sample']:
      torch.manual_seed(0)
      num_sample = {"imdb": 3000, "ag-news": 4000, "sst2": 3000}
      sample_idx = torch.randperm(len(feats))[:num_sample[args.dataset]]
      sampled_feats = feats[sample_idx, :-1].numpy()
      labels = feats[sample_idx, -1].numpy()

    if params['reduce_dim']['method'] == "PCA":
      reducer = KernelPCA(n_components=params['reduce_dim']['dim'], kernel=params['reduce_dim']['kernel'], random_state=0)
      # reducer = PCA(n_components=params['reduce_dim']['dim'], random_state=0)
      # from sklearn.preprocessing import StandardScaler
      # scaler = StandardScaler()
      # sampled_feats = scaler.fit_transform(sampled_feats)
      scaler = None
      reduced_feat = reducer.fit_transform(sampled_feats)
    elif params['reduce_dim']['method'] == 'RF':
      scaler = None
      reducer = RBFSampler(gamma=1, n_components=params['reduce_dim']['dim'], random_state=1)
      reduced_feat = reducer.fit_transform(sampled_feats)
    else:
      assert False, "Not implemented yet. Check json"
  else:
      reducer = None
      scaler = None
      reduced_feat = feats[:, :-1].numpy()
      labels = feats[:, -1].numpy()

  train_stats = get_stats(reduced_feat, labels, use_shared_cov=params['shared_cov'])
  detector = Detector(model_wrapper, args.val_adv, train_stats, logger, params, (scaler, reducer), dataset=args.dataset , seed=args.seed)
  if args.baseline:
    pass
    # detector.test_baseline_PPL(args.test_adv, args.fpr_threshold)
    # detector.test_comb(args.test_adv, args.fpr_threshold)
    # detector.test_baseline(args.test_adv, args.fpr_threshold)
  else:
    roc, auc, tpr_at_fpr, naive_tpr, conf, testset = detector.test(args.test_adv, args.fpr_threshold)

  if args.compute_bootstrap:
    # Compute bootstrap scores
    target = testset.result_type.values
    scores = compute_bootstrap_score(conf, target, roc, args.fpr_threshold)

    logger.log.info("-----Bootstrapped Results-----")
    for k, v in scores.items():
      logger.log.info(f"{k}: {v:.4f}")


