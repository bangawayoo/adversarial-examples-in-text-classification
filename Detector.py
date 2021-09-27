import pdb

import matplotlib.pyplot

from utils.detection import *
from utils.dataset import *
from utils.miscellaneous import *

class Detector():
  def __init__(self, model_wrapper, val_data_path, train_stats, loader,
               logger, params, dim_reducer, dataset=None, seed=0):
    #TODO: max_adv_num should be given as an argumnet depending on the dataset
    self.max_adv_num_dict = {'imdb':2000, 'ag-news':2000, 'sst2':1000}
    self.max_adv_num = self.max_adv_num_dict[dataset]
    self.loader = loader
    self.params = params
    self.model_wrapper = model_wrapper
    self.logger = logger
    self.stats = train_stats
    self.data_path = val_data_path
    self.seed = seed
    self.data = None
    self.batch_size = 32 if dataset=="imdb" else 64
    self.scaler = dim_reducer[0]
    self.dim_reducer = dim_reducer[1]
    self.data = self.get_data(val_data_path, max_adv_num=self.max_adv_num)

  def get_data(self, val_data_path, max_adv_num):
    # max_adv_num : number of maximum adversarial samples to be tested
    # If not possible, this is decremented by 100 until it becomes possible
    if val_data_path.endswith(".csv"):
      dataset, _ = self.loader.get_attack_from_csv(split_type='random_sample', max_adv_num=max_adv_num,
                                         batch_size=128, model_wrapper=None)
    elif val_data_path.endswith(".pkl"):
      dataset = self.loader.get_attack_from_pkl(val_data_path, self.model_wrapper, batch_size=128)

    adv_count = dataset.result_type.value_counts()[1]
    total = len(dataset)
    self.logger.log.info(f"Percentage of adv. samples :{adv_count}/{total} = {adv_count/total:3f}")
    return dataset

  def test(self, test_data_path, fpr_thres):
    testset = self.get_data(test_data_path, max_adv_num=self.max_adv_num)
    texts = testset['text'].tolist()
    gt = torch.tensor(testset['ground_truth_output'].tolist())
    # import random
    # random.seed(0)
    # random.shuffle(texts)
    # label = testset['result_type'].tolist()
    # random.seed(0)
    # random.shuffle(label)
    # testset['result_type'] = label

    test_features, preds = get_test_features(self.model_wrapper, batch_size=self.batch_size, dataset=texts, params=self.params,
                                             logger=self.logger)
    if self.dim_reducer:
      test_features = test_features.numpy()
      if self.scaler:
        test_features = self.scaler.transform(test_features)
      test_features = torch.tensor(self.dim_reducer.transform(test_features))

    confidence, conf_indices, distance = compute_dist(test_features, self.stats, distance_type="euclidean",
                                                      use_marginal=False)

    # confidence = distance[torch.arange(preds.numel()), preds]

    num_nans = sum(confidence == -float("Inf"))
    if num_nans != 0:
      self.logger.log.info(f"Warning : {num_nans} Nans in confidence")
      confidence[confidence == -float("inf")] = -1e6

    metric_header = ["tpr", "fpr", "f1", "auc"]
    self.logger.log.info("-----Results for Euclidean distance------")
    roc, pr, naive_tpr, f1, auc = detect_attack(testset, confidence, fpr_thres, visualize=True, logger=self.logger, mode="euclidean")
    self.logger.save_custom_metric("euclidean", [naive_tpr, fpr_thres, f1, auc], metric_header)

    self.logger.log.info("-----Results for Mahal. OOD------")
    confidence, conf_indices, distance = compute_dist(test_features, self.stats, distance_type="mahal",
                                                      use_marginal=False)
    # confidence = distance[torch.arange(preds.numel()), preds]

    num_nans = sum(confidence == -float("Inf"))
    if num_nans != 0:
      self.logger.log.info(f"Warning : {num_nans} Nans in confidence")
      confidence[confidence == -float("inf")] = -1e6
    roc, pr, tpr_at_fpr, f1, auc = detect_attack(testset, confidence,
                                         fpr_thres,
                                         visualize=True, logger=self.logger, mode="mahal", log_metric=True)
    self.logger.save_custom_metric("mahal", [tpr_at_fpr, fpr_thres, f1, auc], metric_header)

    return roc, auc, tpr_at_fpr, naive_tpr, confidence, testset


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

