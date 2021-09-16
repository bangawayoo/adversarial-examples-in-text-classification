import pdb

import matplotlib.pyplot
import numpy as np

from utils.detection import *
from utils.dataset import *
from utils.miscellaneous import *

class Detector():
  def __init__(self, tune_params, model_wrapper, val_data_path, train_stats,
               logger, params, dim_reducer, kernel_args=None, dataset=None, seed=0):
    #TODO: max_adv_num should be given as an argumnet depending on the dataset
    self.max_adv_num_dict = {'imdb':2000, 'ag-news':2000, 'sst2':1000}
    self.max_adv_num = self.max_adv_num_dict[dataset]
    self.tune_params = tune_params
    self.params = params
    self.model_wrapper = model_wrapper
    self.logger = logger
    self.stats = train_stats
    self.data_path = val_data_path
    self.seed = seed
    self.data = None
    self.best_params = None
    self.batch_size = 32 if dataset=="imdb" else 64
    self.dim_reducer = dim_reducer
    self.kernel_args = kernel_args

    basedir = os.path.join(logger.log_path, 'params')
    self.best_params_path = os.path.join(basedir, f"best_params-seed{self.seed}.txt")
    self.data = self.get_data(val_data_path, max_adv_num=self.max_adv_num)

  def get_data(self, val_data_path, max_adv_num):
    # max_adv_num : number of maximum adversarial samples to be tested
    # If not possible, this is decremented by 100 until possible
    if val_data_path.endswith(".csv"):
      dataset, _ = read_testset_from_csv(val_data_path, use_original=False,
                                                   split_type='random_sample', seed=self.seed, max_adv_num=max_adv_num,
                                         batch_size=128, model_wrapper=None, logger=self.logger)
    elif val_data_path.endswith(".pkl"):
      dataset = read_testset_from_pkl(val_data_path, self.model_wrapper,
                                      batch_size=128, logger=self.logger)

    adv_count = dataset.result_type.value_counts()[1]
    total = len(dataset)
    self.logger.log.info(f"Percentage of adv. samples :{adv_count}/{total} = {adv_count/total:3f}")

    return dataset

  def grid_search(self, fpr_thres, tune_params=False):
    self.logger.log.info("---------Starting Grid Search------------")
    texts = self.data['text'].tolist()
    best = {'tpr':0, 'auc':0}

    start_k, end_k, step_k = self.tune_params['topk']['start'], self.tune_params['topk']['end'], self.tune_params['topk']['step']
    for k in np.arange(start_k, end_k, step_k):
      self.params['prob_param']['topk'] = int(k)
      self.params['prob_param']['p'] = k
      self.params['prob_param']['ratio'] = k
      self.logger.log.info(f"K={k}")
      test_features, probs = get_test_features(self.model_wrapper, batch_size=self.batch_size, dataset=texts, params=self.params,
                                               logger=self.logger)
      confidence, conf_indices, distance = compute_dist(test_features, self.stats, distance_type="euclidean",
                                                        use_marginal=False)

      if probs.dim() == 1:
        probs = probs.unsqueeze(-1)
      num_nans = sum(probs == -float("Inf"))
      if num_nans != 0:
        self.logger.log.info(f"Warning : {num_nans} Nans in conditional probability")
        probs[probs == -float("inf")] = -1e6
      refined_confidence = confidence + probs.squeeze()
      refined_confidence[refined_confidence == -float("Inf")] = -1e6
      refined_confidence[torch.isnan(refined_confidence)] = -1e6

      self.logger.log.info("-----Results for Baseline OOD------")
      roc, pr, tpr, f1, auc = detect_attack(self.data, confidence, fpr_thres,
                                  visualize=False, logger=self.logger, mode="Baseline")
      self.logger.log.info("-----Results for Hierarchical OOD------")
      roc, pr, tpr_at_fpr, f1, auc = detect_attack(self.data, refined_confidence,
                                           fpr_thres,
                                           visualize=False, logger=self.logger, mode="Hierarchical")
      self.logger.log.info("-----Results for Conditional Probability OOD------")
      _, _, _, _, _ = detect_attack(self.data, probs, fpr_thres,
                              visualize=False, logger=self.logger, mode="conditional")

      if auc > best['auc']:
        best['tpr'] = tpr_at_fpr
        best['k'] = k
        best['roc'] = roc
        best['auc'] = auc
        best_conf = refined_confidence

    self.best_params = best['k']
    self.logger.log.info(f"Best : {best['auc']} at k={best['k']}")
    self.save_best_params()

    return self.best_params

  def save_best_params(self):
    best_params_str = str(self.best_params)
    save_txt(best_params_str, self.best_params_path)

  def test(self, test_data_path, fpr_thres):
    testset = self.get_data(test_data_path, max_adv_num=self.max_adv_num)
    texts = testset['text'].tolist()

    # assert self.best_params is not None, "Check if params is tuned"
    if self.best_params is not None:
      self.logger.log.info(f"---------Test Mode with k = {self.best_params}---------")
      self.params['prob_param']['topk'] = int(self.best_params)
      self.params['prob_param']['p'] = self.best_params
      self.params['prob_param']['ratio'] = self.best_params
    test_features, probs = get_test_features(self.model_wrapper, batch_size=self.batch_size, dataset=texts, params=self.params,
                                             logger=self.logger)
    if self.dim_reducer:
      test_features = torch.tensor(self.dim_reducer.transform(test_features.numpy()))

    confidence, conf_indices, distance = compute_dist(test_features, self.stats, distance_type="euclidean",
                                                      use_marginal=False)


    uncertainty = []
    GPs = self.kernel_args[0]
    for gp, stats in zip(GPs, self.stats):
      mu = stats[0]
      test_input = test_features.numpy() - mu
      std = gp.predict(test_input, return_std=True)[1][:,None]
      uncertainty.append(std)

    uncertainty = np.concatenate(uncertainty, axis=-1)
    probs = -torch.tensor([v[idx] for v, idx in zip(uncertainty, conf_indices)])
    print(probs.mean(), probs.std())

    num_nans = sum(probs == -float("Inf"))
    if num_nans != 0:
      self.logger.log.info(f"Warning : {num_nans} Nans in conditional probability")
      probs[probs == -float("inf")] = 1e-3

    probs = (probs - probs.mean()) / probs.std()
    confidence = (confidence - confidence.mean()) / confidence.std()
    for weight in [1]:
      # print("Using Weight ", weight)
      refined_confidence = weight * confidence + probs.squeeze()
      refined_confidence[refined_confidence == -float("Inf")] = -1e6
      refined_confidence[torch.isnan(refined_confidence)] = -1e6

      metric_header = ["tpr", "fpr", "f1", "auc"]
      self.logger.log.info("-----Results for Baseline OOD------")
      roc, pr, naive_tpr, f1, auc = detect_attack(testset, confidence, fpr_thres, visualize=True, logger=self.logger, mode="Baseline")
      self.logger.save_custom_metric("baseline_OOD", [naive_tpr, fpr_thres, f1, auc], metric_header)

      self.logger.log.info("-----Results for Hierarchical OOD------")
      roc, pr, tpr_at_fpr, f1, auc = detect_attack(testset, refined_confidence,
                                           fpr_thres,
                                           visualize=True, logger=self.logger, mode="Hierarchical", log_metric=True)
      self.logger.save_custom_metric("results", [tpr_at_fpr, fpr_thres, f1, auc, self.best_params], metric_header+["topk"])

      self.logger.log.info("-----Results for Conditional Probability OOD------")
      roc, pr, tpr_at_fpr, f1, auc = detect_attack(testset, probs, fpr_thres,
                              visualize=True, logger=self.logger, mode="conditional")
      self.logger.save_custom_metric("token-cond_prob", [tpr_at_fpr, fpr_thres, f1, auc], metric_header)
    return roc, auc, tpr_at_fpr, naive_tpr, refined_confidence, testset


  def test_baseline(self, test_data_path, fpr_thres):
    testset = self.get_data(test_data_path, max_adv_num=2000)
    texts = testset['text'].tolist()

    self.logger.log.info("---------Baseline Test Mode---------")
    max_probs, negative_entropy = get_softmax(self.model_wrapper, batch_size=self.batch_size, dataset=texts,
                                             logger=self.logger)

    num_nans = sum(negative_entropy == -float("Inf"))
    if num_nans != 0:
      self.logger.log.info(f"Warning : {num_nans} Nans in entropy")
      negative_entropy[negative_entropy == -float("inf")] = -1e6

    metric_header = ["tpr", "fpr", "f1", "auc"]
    self.logger.log.info("-----Results for Baseline: Max. probability------")
    roc, pr, tpr, f1, auc = detect_attack(testset, max_probs, fpr_thres,
                                visualize=False, logger=self.logger, mode="Baseline:MaxProb", log_metric=True)
    self.logger.save_custom_metric("max_prob-results", [tpr, fpr_thres, f1, auc], metric_header)

    self.logger.log.info("-----Results for Baseline: negative entropy------")
    roc, pr, tpr, f1, auc = detect_attack(testset, negative_entropy,
                                         fpr_thres,
                                         visualize=False, logger=self.logger, mode="Baseline:NegEnt.", log_metric=True)
    self.logger.save_custom_metric("neg_ent-results", [tpr, fpr_thres, f1, auc], metric_header)
    return

  def test_baseline_PPL(self, test_data_path, fpr_thres):
    testset = self.get_data(test_data_path, max_adv_num=2000)
    texts = testset['text'].tolist()
    confidence = compute_ppl(texts)
    metric_header = ["tpr", "fpr", "f1", "auc"]
    self.logger.log.info("-----Results for Baseline: GPT-2 PPL------")
    roc, pr, tpr, f1, auc = detect_attack(testset, confidence, fpr_thres,
                                visualize=False, logger=self.logger, mode="Baseline:PPL", log_metric=True)
    self.logger.save_custom_metric("ppl", [tpr, fpr_thres, f1, auc], metric_header)

