import pickle
import random

from sklearn.metrics import roc_auc_score
import numpy as np


def save_pkl(object, path):
  with open(path, "wb") as handle:
    pickle.dump(object, handle)

def load_pkl(path):
  with open(path, "rb") as handle:
    object = pickle.load(handle)
  return object

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