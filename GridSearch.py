from utils.detection import *
from utils.dataset import *
from utils.miscellaneous import *

class GridSearch():
  def __init__(self, tune_params, model_wrapper, val_data_path, train_stats,
               logger, params, seed=0):
    self.tune_params = tune_params
    self.params = params
    self.model_wrapper = model_wrapper
    self.logger = logger
    self.stats = train_stats
    self.data_path = val_data_path
    self.seed = 0
    self.data = None
    self.best_params = None

    basedir = os.path.dirname(self.data_path)
    self.best_params_path = os.path.join(basedir, f"best_params-seed{self.seed}.pkl")

    self.data = self.get_data(val_data_path)

  def get_data(self, val_data_path):
    if val_data_path.endswith(".csv"):
      testset, _ = read_testset_from_csv(val_data_path, use_original=False,
                                                   split_type='random_sample', seed=self.seed)
    elif val_data_path.endswith(".pkl"):
      testset = read_testset_from_pkl(val_data_path, self.model_wrapper,
                                      batch_size=128, logger=self.logger )
    return testset

  def tune(self, fpr_thres):
    if os.path.exists(self.best_params_path):
      self.best_params = load_pkl(self.best_params_path)
      self.logger.log.info(f"Using Existing param in {self.best_params_path}")
      return self.best_params

    self.logger.log.info("---------Starting Grid Search------------")
    texts = self.data['text'].tolist()
    best = {'tpr':0}

    start_k, end_k, step_k = self.tune_params['topk']['start'], self.tune_params['topk']['end'], self.tune_params['topk']['step']
    for k in range(start_k, end_k, step_k):
      self.params['prob_param']['topk'] = k
      self.logger.log.info(f"-----K={k}------")
      test_features, probs = get_test_features(self.model_wrapper, batch_size=128, dataset=texts, params=self.params,
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
      roc, auc, _ = detect_attack(self.data, confidence, conf_indices, fpr_thres,
                                  visualize=False, logger=self.logger, mode="Baseline")
      self.logger.log.info("-----Results for Hierarchical OOD------")
      roc, auc, tpr_at_fpr = detect_attack(self.data, refined_confidence, conf_indices,
                                           fpr_thres,
                                           visualize=False, logger=self.logger, mode="Hierarchical")
      self.logger.log.info("-----Results for Conditional Probability OOD------")
      _, _, _ = detect_attack(self.data, probs, conf_indices, fpr_thres,
                              visualize=False, logger=self.logger, mode="conditional")
      if tpr_at_fpr > best['tpr']:
        best['tpr'] = tpr_at_fpr
        best['k'] = k
        best['roc'] = roc
        best_conf = refined_confidence

      self.best_params = best
      self.logger.log.info(f"Best : {best['tpr']} at p={best['k']}")
      self.save_best_params()

      return self.best_params

  def save_best_params(self):
    save_pkl(self.best_params, self.best_params_path)

  def test(self, test_data_path, fpr_thres):
    testset = self.get_data(test_data_path)
    texts = testset['text'].tolist()
    assert self.best_params is not None, "Check if params is tuned"

    self.logger.log.info(f"---------Test Mode with k = {self.best_params['k']}---------")
    self.params['prob_param']['topk'] = self.best_params['k']
    test_features, probs = get_test_features(self.model_wrapper, batch_size=128, dataset=texts, params=self.params,
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
    roc, auc, _ = detect_attack(testset, confidence, conf_indices, fpr_thres,
                                visualize=True, logger=self.logger, mode="Baseline")
    self.logger.log.info("-----Results for Hierarchical OOD------")
    roc, auc, tpr_at_fpr = detect_attack(testset, refined_confidence, conf_indices,
                                         fpr_thres,
                                         visualize=True, logger=self.logger, mode="Hierarchical")
    self.logger.log.info("-----Results for Conditional Probability OOD------")
    _, _, _ = detect_attack(testset, probs, conf_indices, fpr_thres,
                            visualize=True, logger=self.logger, mode="conditional")

    return roc, auc, tpr_at_fpr, refined_confidence, testset