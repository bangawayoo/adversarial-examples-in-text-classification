import pdb

from utils.detection import *
from utils.dataset import *
from utils.miscellaneous import *

class Detector():
  def __init__(self, model_wrapper, train_stats, loader,
               logger, params, modules, use_val=False, dataset=None, seed=0):

    self.loader = loader
    self.params = params
    self.model_wrapper = model_wrapper
    self.logger = logger
    self.stats = train_stats
    self.seed = seed
    self.data = None
    self.dtype = 'val' if use_val else 'test'
    self.batch_size = 32 if dataset=="imdb" else 64
    self.scaler = modules[0]
    self.dim_reducer = modules[1]
    self.estimators = modules[2]
    self.estimator_name = modules[3]

  def get_data(self, pkl_path=None):
    if pkl_path and pkl_path.endswith(".pkl"):
      dataset = self.loader.get_attack_from_pkl(pkl_path, self.model_wrapper, batch_size=128)
    else:
      dataset, _ = self.loader.get_attack_from_csv(batch_size=128, dtype=self.dtype, model_wrapper=None)

    adv_count = dataset.result_type.value_counts()[1]
    total = len(dataset)
    self.logger.log.info(f"Percentage of adv. samples :{adv_count}/{total} = {adv_count/total:3f}")
    return dataset

  def test(self, fpr_thres, pkl_path=None):
    testset = self.get_data(pkl_path)
    texts = testset['text'].tolist()
    att_result = torch.tensor(testset['result_type'].tolist())
    path_to_gt = os.path.join(self.logger.log_path, "gt.csv")
    if True:
      save_array(att_result, path_to_gt, append=False)

    test_features, preds = get_test_features(self.model_wrapper, batch_size=self.batch_size, dataset=texts, params=self.params,
                                             logger=self.logger)
    if self.dim_reducer:
      test_features = test_features.numpy()
      if self.scaler:
        test_features = self.scaler.transform(test_features)
      test_features = torch.tensor(self.dim_reducer.transform(test_features))

    metric_header = ["tpr", "fpr", "f1", "auc"]
    path_to_conf = os.path.join(self.logger.log_path, "conf.csv")

    for name, stats, estim in zip(["MLE", self.estimator_name], self.stats, self.estimators):
      self.logger.log.info("-----Results-----")
      self.logger.log.info(f"Using {name} estimator")
      if estim:
        all_confidences = []
        test_features = test_features.numpy()
        for per_cls_estim in estim:
          dist = per_cls_estim.mahalanobis(test_features).reshape(-1,1)
          all_confidences.append(dist)
        all_confidences = np.concatenate(all_confidences, axis=1)
        confidence = -torch.tensor(all_confidences[np.arange(preds.numel()), preds])
      else:
        confidence, conf_indices, conf_all = compute_dist(test_features, stats, use_marginal=False)
        confidence = conf_all[torch.arange(preds.numel()), preds]

      if True:
        save_array(confidence, path_to_conf)

      num_nans = sum(confidence == -float("Inf"))
      if num_nans != 0:
        self.logger.log.info(f"Warning : {num_nans} Nans in confidence")
        confidence[confidence == -float("inf")] = -1e6
      roc, pr, tpr_at_fpr, f1, auc = detect_attack(testset, confidence,
                                           fpr_thres,
                                           visualize=True, logger=self.logger, mode=f"{name}-estim", log_metric=True)
      self.logger.save_custom_metric(f"{name}-estim.", [tpr_at_fpr, fpr_thres, f1, auc], metric_header)

    return roc, auc, tpr_at_fpr, confidence, testset


  def test_baseline_PPL(self, fpr_thres, pkl_path=None):
    testset = self.get_data()
    texts = testset['text'].tolist()
    confidence = compute_ppl(texts)
    confidence[torch.isnan(confidence)] = 1e6
    confidence[confidence == -float("inf")] = -1e6
    metric_header = ["tpr", "fpr", "f1", "auc"]
    self.logger.log.info("-----Results for Baseline: GPT-2 PPL------")
    roc, pr, tpr, f1, auc = detect_attack(testset, confidence, fpr_thres,
                                visualize=False, logger=self.logger, mode="Baseline:PPL", log_metric=True)
    self.logger.save_custom_metric("ppl", [tpr, fpr_thres, f1, auc], metric_header)

