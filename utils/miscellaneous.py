import pdb
import pickle
import random

from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf, MinCovDet, GraphicalLasso, OAS
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

import numpy as np
import os

def save_pkl(object, path):
  dirname = os.path.dirname(path)
  if not os.path.exists(dirname):
    os.mkdir(dirname)
  with open(path, "wb") as handle:
    pickle.dump(object, handle)

def load_pkl(path):
  with open(path, "rb") as handle:
    object = pickle.load(handle)
  return object

def load_txt(path):
  with open(path, "r") as handle:
    object = handle.readline()
  return object

def save_txt(object, path):
  dirname = os.path.dirname(path)
  if not os.path.exists(dirname):
    os.mkdir(dirname)
  with open(path, "w") as handle:
    handle.write(object)

def save_array(object, path, append=True):
  if os.path.exists(path) and append:
    existing_data = np.loadtxt(path)
    if len(existing_data.shape) == 1 :
      existing_data = np.expand_dims(existing_data, -1)
    if len(object.shape) == 1:
      object = np.expand_dims(object, -1)
    existing_data = np.concatenate([existing_data, object], axis=-1)
    np.savetxt(path, existing_data)
    return
  else:
    np.savetxt(path, object)



def bootstrap_sample(all_unperturbed, all_perturbed, bootstrap_sample_size=2000):
  """
  Adapted from repository of "Mozes, Maximilian, et al. "Frequency-guided word substitutions for detecting textual adversarial examples." EACL (2021)."
  all_~ : list of lists with (negative_conf, target)
  """
  scores_sum = {}
  perturbed_auc_scores = [score for score, _ in all_perturbed]
  perturbed_auc_labels = [1] * len(perturbed_auc_scores)
  unperturbed_auc_labels = [0] * len(perturbed_auc_scores)
  pos = len(all_perturbed)
  t_p = [l for _, l in all_perturbed].count(1)
  f_n = pos - t_p

  for _ in range(bootstrap_sample_size):
    neg = pos
    sample = random.sample(all_unperturbed, neg)
    f_p = [l for _, l in sample].count(1)
    t_n = neg - f_p
    unperturbed_auc_scores = [score for score, _ in sample]

    scores = compute_scores(
      perturbed_auc_scores + unperturbed_auc_scores,
      perturbed_auc_labels + unperturbed_auc_labels,
      pos,
      neg,
      t_p,
      t_n,
      f_p,
      f_n,
    )

    for name, score in scores.items():
      try:
        scores_sum[name].append(score)
      except KeyError:
        scores_sum[name] = [score]

  return scores_sum


def compute_scores(probs_one, labels, pos, neg, t_p, t_n, f_p, f_n, round_scores=False):
  """
  Adapted from repository of "Mozes, Maximilian, et al. "Frequency-guided word substitutions for detecting textual
  adversarial examples." EACL (2021)."
  """
  assert t_p + f_n == pos
  assert t_n + f_p == neg
  assert len(probs_one) == pos + neg == len(labels)

  scores = {
      "auc": roc_auc_score(labels, probs_one),
      "tpr": t_p / pos if pos > 0 else 0,
      "fpr": f_p / neg if neg > 0 else 0,
      "tnr": t_n / neg if neg > 0 else 0,
      "fnr": f_n / pos if pos > 0 else 0,
      "pr": t_p / (t_p + f_p) if t_p + f_p > 0 else 0,
      "re": t_p / (t_p + f_n) if t_p + f_n > 0 else 0,
      "f1": (2 * t_p) / (2 * t_p + f_p + f_n) if 2 * t_p + f_p + f_n > 0 else 0,
      "acc": (t_p + t_n) / (pos + neg) if pos + neg > 0 else 0,
  }

  if round_scores:
      scores = {k: np.round(v * 100, 1) for k, v in scores.items()}

  return scores

def compute_bootstrap_score(confidence, target, roc, fpr_thres):
  fpr, tpr, thres1 = roc
  mask = (fpr <= fpr_thres)
  roc_cutoff = np.sort(np.unique(mask*thres1))[1]
  pred = -np.array(confidence) >= roc_cutoff
  pred = pred.astype('int')

  idx = (target==1)
  perturb_pred = pred[idx]
  perturbed = confidence[idx].tolist()
  perturbed = [[-c,p] for c, p in zip(perturbed, perturb_pred)]

  idx = (target==0)
  unperturbed = confidence[idx].tolist()
  unperturb_pred = pred[idx]
  unperturbed = [[-c,p] for c, p in zip(unperturbed, unperturb_pred)]

  scores_sum = bootstrap_sample(unperturbed, perturbed, bootstrap_sample_size=2000)
  scores = {k: np.mean(v) for k, v in scores_sum.items()}
  return scores


def return_cov_estimator(name, params):
  if name == 'OAS':
    return OAS()
  elif name == 'MCD':
    return MinCovDet(support_fraction=params['h'] if params.get("h", None) else None)
  elif name == 'ledoit-wolf':
    return LedoitWolf()
  else:
    return None


def preprocess_features(feats, params, args, logger):
  scaler = StandardScaler() if params.get('scaler', None) else None
  if params['sample']:
    np.random.seed(0)
    num_sample = {"imdb": 8000, "ag-news": 8000, "sst2": 8000}
    sample_idx = np.random.choice(range(len(feats)), size=num_sample[args.dataset], replace=False)
    sampled_feats = feats[sample_idx, :-1]
    labels = feats[sample_idx, -1]
  else:
    sampled_feats = feats[:, :-1]
    labels = feats[:, -1]

  if params['reduce_dim']['do']:
    if params['reduce_dim']['method'] == "PCA":
      reduced_feat, reducer = return_PCA_features(sampled_feats, params, scaler, logger, seed=0)
    elif params['reduce_dim']['method'] == 'RF':
      scaler = None
      reducer = RBFSampler(gamma=1, n_components=params['reduce_dim']['dim'], random_state=0)
      reduced_feat = reducer.fit_transform(sampled_feats)
    else:
      assert False, "Not implemented yet. Check json"
  else:
    reducer = None
    reduced_feat = scaler.fit_transform(sampled_feats) if scaler else sampled_feats

  return reduced_feat, labels, reducer, scaler

def return_PCA_features(feats, params, scaler, logger, seed=0):
  if params['reduce_dim']['dim'] < 1:  # Determine number of components by explained variance
    tmp_reducer = KernelPCA(n_components=feats.shape[-1], gamma=(1 / feats.shape[-1]), \
                            kernel=params['reduce_dim']['kernel'], random_state=seed)
    if scaler:
      feats = scaler.fit_transform(feats)
    tmp_reducer.fit(feats)
    ev = tmp_reducer.lambdas_
    ev_cumsum = np.cumsum(ev / ev.sum())
    n_components = sum(ev_cumsum < params['reduce_dim']['dim'])
    logger.log.info(f"Using {n_components} components to explain {params['reduce_dim']['dim']}")
    reducer = KernelPCA(n_components=n_components, gamma=(1 / feats.shape[-1]), \
                        kernel=params['reduce_dim']['kernel'], random_state=seed)
  else:  # Choose number of components by fixed number
    reducer = KernelPCA(n_components=params['reduce_dim']['dim'], gamma=(1 / feats.shape[-1]), \
                        kernel=params['reduce_dim']['kernel'], random_state=seed)
    # reducer = PCA(n_components=0.9, random_state=0)

  if scaler:
    sampled_feats = scaler.fit_transform(feats)
  reduced_feat = reducer.fit_transform(feats)
  return reduced_feat, reducer