import argparse
from textattack import attack_results
import pandas as pd
import torch

parser = argparse.ArgumentParser(description="Detect and defense attacked samples")

parser.add_argument("--dataset", default="imdb", type=str,
                    choices=["dbpedia14", "ag_news", "imdb", "yelp", "mnli", "sst2"],
                    help="classification dataset to use")
parser.add_argument("--mnli_option", default="matched", type=str,
                    choices=["matched", "mismatched"],
                    help="use matched or mismatched test set for MNLI")
parser.add_argument("--target_model", default="textattack/roberta-base-imdb", type=str,
                    help="type of model")
parser.add_argument("--adv_from_file", default="attack_results/roberta-base-imdb_pwws.csv", type=str,
                    help="perturbed texts from csv")
parser.add_argument("--split_ratio", default=0.9)

parser.add_argument("--gpu", default='1', type=str)

args, _ = parser.parse_known_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from utils.inference import *
from utils.detection import *
from utils.dataset import *
from utils.preprocess import *

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

#List of hyper-parameters
REFINE_TOP_K=6
ADV_RATIO=None
LAYER = -1

if __name__ == "__main__":
  model = AutoModelForSequenceClassification.from_pretrained(args.target_model, output_hidden_states=True).cuda()
  ckpt = torch.load("fgws_ckpt/model.pth")['model_state_dict']
  model.load_state_dict(ckpt)
  model.eval()
  tokenizer = AutoTokenizer.from_pretrained('roberta-base', do_lower_case=True)

  trainvalset, _, key = get_dataset(args)
  text_key, testset_key = key
  trainset, valset = split_dataset(trainvalset, split='trainval', split_ratio=args.split_ratio)
  trainset[text_key] = [fgws_preprocess(sample.split()) for sample in trainset[text_key]]
  train_features = get_train_features(model, tokenizer, args, batch_size=512, dataset=trainset, text_key=text_key, layer=LAYER)
  train_stats = get_stats(train_features, use_shared_cov=True)

  # testset, raw_testset = read_testset_from_csv(args.adv_from_file, use_original=False, split_type='fgws', split_ratio=0.6, seed=2)  # returns pandas dataframe
  testset = read_testset_from_pkl("prioritized-adv_examples.pkl", model, tokenizer)  # returns pandas dataframe
  total, adv_count = testset.result_type.value_counts().sum(), testset.result_type.value_counts()[1]
  print(f"Adv success rate {adv_count}/{total} : {adv_count / total}")

  texts = testset['text'].tolist()
  print("Building test features...")
  test_features, probs = get_test_features(model, tokenizer, batch_size=128, dataset=texts, topk=10, layer=LAYER, use_cosine_sim=False)
  confidence, conf_indices, distance = compute_dist(test_features, train_stats, distance_type="euclidean", use_marginal=False)

  conf_indices = torch.max(distance + probs.unsqueeze(-1), dim=1).indices
  gt = torch.tensor(testset.loc[testset['result_type']==0, 'ground_truth_output'].values)
  correct = conf_indices[(testset.result_type==0).values].eq(gt).sum()
  print(f"Accuracy of hard clustering on {correct}/{gt.numel()}: {correct/gt.numel()}")
  # Standerdize probs and confidence
  # probs = (probs - torch.mean(probs)) / torch.std(probs)
  # confidence = (confidence - torch.mean(confidence)) / torch.std(confidence)

  num_nans = probs[probs==-float("Inf")].sum()
  if num_nans > 0 :
    print(f"Warning : {num_nans} Nans in conditional probability")
  refined_confidence = confidence + probs.squeeze()
  refined_confidence[refined_confidence==-float("Inf")] = -1e6
  refined_confidence[torch.isnan(refined_confidence)] = -1e6

  # Detect attacks for correctly classified samples
  fpr_thres = 0.093
  adv_count = testset.loc[testset['result_type']==1].shape[0]
  correct_idx = np.array(testset['result_type']!=-1)
  correct_set = testset.loc[correct_idx]
  #TODO: Calcuate precision, recall
  roc, auc = detect_attack(correct_set, confidence[correct_idx], conf_indices[correct_idx], fpr_thres, visualize=False, by_class=False)
  roc, auc = detect_attack(correct_set, refined_confidence[correct_idx], conf_indices[correct_idx], fpr_thres, visualize=False, by_class=False)

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

