import math
import os
import pdb
import pickle
import random
import re

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, precision_recall_fscore_support, auc

from utils.miscellaneous import save_pkl, load_pkl


# Forward using train data to get empirical mean and covariance
def get_stats(features, use_shared_cov=False):
  # Compute mean and covariance of each cls.
  stats = []
  label_list = range(len(features))

  if use_shared_cov :
    shared_cov = None
    shared_feat = []

    for idx, lab in enumerate(label_list):
      feat = torch.cat(features[idx])
      shared_feat.append(feat)
      feat = feat.numpy()
      mu = feat.mean(axis=0)
      stats.append([mu, 0])

    shared_feat = torch.cat(shared_feat).numpy()
    shared_cov = np.cov(shared_feat.T)

    for idx, lab in enumerate(label_list):
      stats[idx][1] = shared_cov

    return stats
  else:
    for idx, lab in enumerate(label_list):
      feat = torch.cat(features[idx])
      feat = feat.numpy()
      mu = feat.mean(axis=0)
      cov = np.cov(feat.T)
      stats.append([mu, cov])


  return stats


def get_train_features(model_wrapper, args, batch_size, dataset, text_key, layer=-1):
  assert layer=='pooled' or layer < 0 , "Layer either has to be a int between -12~-1 or the pooling layer"
  model_name = os.path.basename(args.target_model)
  model_name += f"-layer_{layer}"

  if os.path.exists(f"saved_feats/{model_name}.pkl"):
    features = load_pkl(f"saved_feats/{model_name}.pkl")
    return features

  print("Building train features")
  model = model_wrapper.model
  num_samples = len(dataset['label'])
  label_list = np.unique(dataset['label'])
  num_labels = len(label_list)
  num_batches = int((num_samples // batch_size) + 1)
  features = [[] for _ in range(num_labels)]

  # num_batches = 2
  with torch.no_grad():
    for i in tqdm(range(num_batches)):
      lower = i * batch_size
      upper = min((i + 1) * batch_size, num_samples)
      examples = dataset[text_key][lower:upper]
      labels = dataset['label'][lower:upper]
      y = torch.LongTensor(labels)
      output = model_wrapper.inference(examples, output_hs=True)
      preds = output.logits
      if type(layer) == int:
        feat = output.hidden_states[layer][:, 0, :].cpu()  # (Batch_size, 768)
        for idx, lab in enumerate(label_list):
          features[idx].append(feat[y == lab])
      elif layer == 'pooled':
        feat = output.hidden_states[-1]  # (Batch_size, 768)
        pooled_feat = model.bert.pooler(feat).cpu()
        for idx, lab in enumerate(label_list):
          features[idx].append(pooled_feat[y==lab])

  save_pkl(features, f"saved_feats/{model_name}.pkl")

  return features


def get_test_features(model_wrapper, batch_size, dataset, params, logger=None):
  # dataset, batch_size, i, layer = testset['text'].tolist(), 32, 0, -1
  assert logger is not None, "No logger given"
  num_samples = len(dataset)
  num_batches = int((num_samples // batch_size) + 1)
  features = []
  probs = []
  layer = params['layer_param']['cls_layer']

  with torch.no_grad():
    for i in tqdm(range(num_batches)):
      lower = i * batch_size
      upper = min((i + 1) * batch_size, num_samples)
      examples = dataset[lower:upper]
      if len(examples) == 0:
        continue
      output = model_wrapper.inference(examples, output_hs=True, output_attention=True)
      # logit = output.logits
      # prob = F.softmax(logit, dim=-1)
      att_mask = model_wrapper.get_att_mask(examples)

      if type(layer) == int:
        feat = output.hidden_states[layer][:, 0, :].cpu()  # output.hidden_states : (Batch_size, sequence_length, hidden_dim)
        features.append(feat.cpu())
        log_prob_sum = torch.zeros(len(examples)).to(model_wrapper.device)
        start_seq = 1 if params['model_param']['exclude_cls_token'] else 0
        for l in range(1, params['layer_param']['num_layer']+1):
          if params['model_param']['type'] == "cosine_sim":
            query = output.hidden_states[layer-l+1][:, 0, :].to(model_wrapper.device) #(batch_size, hidden_dim)
            word_vectors = output.hidden_states[layer-l][:, start_seq:, :]#(batch_size, seq  , hidden_dim)
            word_vectors = word_vectors / torch.norm(word_vectors, p=2, dim=-1, keepdim=True)
            word_vectors = word_vectors * att_mask[:,start_seq:,None].to(model_wrapper.device)
            query = query / torch.norm(query, p=2, dim=-1, keepdim=True)
            inner_product = torch.einsum("bh, bsh -> bs", query, word_vectors)
            seq_len = att_mask.sum(-1).long().squeeze()
            for idx, seq in enumerate(seq_len):
              inner_product[idx, seq:] = 0
            if params['model_param']['normalization'] == 'l1':
              inner_product[inner_product <= 0] = 1e-12
              conditional_prob = inner_product / torch.sum(inner_product, dim=-1, keepdim=True)
            elif params['model_param']['normalization'] == 'softmax':
              T = params['model_param']['temperature']
              conditional_prob = torch.exp(inner_product/T) / torch.exp(inner_product/T).sum(-1, keepdims=True)


          elif params['model_param']['type'] == "attention":
            start_seq = 1 if params['model_param']['exclude_cls_token'] else 0
            if params['model_param']['attention_type'] == "key":
              conditional_prob = output.attentions[layer-l+1][:,:,start_seq:,0] #attention score using cls token as key. Then, sum across heads
            elif params['model_param']['attention_type'] == "query":
              conditional_prob = output.attentions[layer-l+1][:,:,0,start_seq:] #attention score using cls token as query. Then, sum across heads
            if params['prob_param']['sum_heads']:
              conditional_prob = conditional_prob.sum(1) # Sum across heads
            else:
              bs = conditional_prob.shape[0]
              conditional_prob = conditional_prob.reshape(bs, -1)

            seq_len = att_mask.sum(-1).long().squeeze()
            for idx, seq in enumerate(seq_len):
              conditional_prob[idx, seq:] = 0
            conditional_prob = conditional_prob / conditional_prob.sum(-1, keepdim=True)  #Normalize across sequences

          if params['prob_param']['choose_type'] == 'topk':
            if conditional_prob.shape[-1] < params['prob_param']['topk']:
              params['prob_param']['topk'] = conditional_prob.shape[-1]
              logger.log.debug(f"Topk param truncated to {conditional_prob.shape[-1]}")
            topk_prob = torch.topk(conditional_prob, params['prob_param']['topk'], dim=-1).values #Top k values across sequences
            topk_prob[topk_prob==0] = 1
            log_prob_layer = torch.sum(torch.log(topk_prob), dim=-1)
          elif params['prob_param']['choose_type'] == 'ratio':
            token_prob = conditional_prob
            sorted, indices = torch.sort(token_prob, dim=-1, descending=True)
            seq_len_ratio = (att_mask.sum(-1) * params['prob_param']['ratio']).long().squeeze()
            for idx, seq in enumerate(seq_len_ratio):
              sorted[idx, seq:] = 1
            log_prob_layer = torch.sum(torch.log(sorted), dim=-1)
          elif params['prob_param']['choose_type'] == 'threshold':
            token_prob = conditional_prob
            sorted, indices = torch.sort(token_prob, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted, dim=-1)
            sorted[cumsum > params['prob_param']['p']] = 1
            log_prob_layer = torch.sum(torch.log(sorted), dim=-1)

            # if not params['prob_param']['sum_heads']:
            #   token_prob = token_prob.sum(1)
            #   token_prob = token_prob / token_prob.sum(-1, keepdim=True) # Sum across top k sequences
            #   log_prob_layer = torch.sum(torch.log(token_prob), dim=-1)

          # Add each layer's log prob.
          log_prob_sum += log_prob_layer.to(model_wrapper.device)
        probs.append(log_prob_sum.cpu())

      else:
        feat = output.hidden_states[-1]  # (Batch_size, 768)
        pooled_feat = model_wrapper.model.bert.pooler(feat)
        features.append(pooled_feat.cpu())
        probs.append(torch.empty(pooled_feat.shape[0],1))

  return torch.cat(features, dim=0), torch.cat(probs, dim=0)


def get_softmax(model_wrapper, batch_size, dataset, logger=None):
  assert logger is not None, "No logger given"
  num_samples = len(dataset)
  num_batches = int((num_samples // batch_size) + 1)
  probs = []
  negative_entropy = []

  with torch.no_grad():
    for i in tqdm(range(num_batches)):
      lower = i * batch_size
      upper = min((i + 1) * batch_size, num_samples)
      examples = dataset[lower:upper]
      if len(examples) == 0:
        continue
      output = model_wrapper.inference(examples, output_hs=True, output_attention=True)
      logit = output.logits
      prob = F.softmax(logit, dim=-1)
      max_prob = torch.max(prob, dim=-1).values
      neg_ent = F.softmax(logit, dim=-1) * F.log_softmax(logit, dim=1)
      neg_ent = neg_ent.sum(-1)
      probs.append(max_prob.cpu())
      negative_entropy.append(neg_ent.cpu())

  return torch.cat(probs, dim=0), torch.cat(negative_entropy, dim=0)


def compute_dist(test_features, train_stats, distance_type='mahan', use_marginal=True):
  # stats is list of np.array ; change to torch operations for gradient update
  output = []
  if distance_type == "mahan":
    print("Using mahanalobis distance...")
    for (mu, cov) in train_stats:
      mu, cov = torch.tensor(mu).double(), torch.tensor(cov).double()
      prec = torch.inverse(cov)
      delta = test_features-mu
      neg_dist = - torch.einsum('nj, jk, nk -> n', delta, prec, delta)
      log_likelihood = (0.5 * neg_dist) + math.log((2 * math.pi) ** (-mu.shape[0] / 2))
      output.append(log_likelihood.unsqueeze(-1))
  elif distance_type == "euclidean":
    print("Using euclidean distance...")
    for (mu, cov) in train_stats:
      mu = torch.tensor(mu).double()
      delta = test_features-mu
      neg_dist = - torch.norm(delta, p=2, dim=1)**2
      log_likelihood = (0.5 * neg_dist) + math.log((2*math.pi)**(-mu.shape[0]/2))
      output.append(log_likelihood.unsqueeze(-1))

  output = torch.cat(output, dim=1)
  confidence, conf_indices = torch.max(output, dim=1) # Takes the max of class conditional probability
  if use_marginal:
    confidence = torch.log(torch.sum(torch.exp(output), dim=1)) # Takes the marginal probability
  return confidence, conf_indices, output


def detect_attack(testset, confidence, fpr_thres=0.05, visualize=False, logger=None, mode=None,
                  log_metric=False):
  """
  Detect attack for correct samples only to compute detection metric (TPR, recall, precision)
  """
  assert logger is not None, "Logger not given"
  # adv_count=None; visualize=True; fpr_thres=0.05
  target = np.array(testset['result_type'].tolist())
  # target[target==-1] = 1
  conf = confidence.numpy()
  testset['negative_conf'] = -conf # negative of confidence : likelihood of adv. probability

  # Class-agnostic
  fpr, tpr, thres1 = roc_curve(target, -conf)
  precision, recall, thres2 = precision_recall_curve(target, -conf)
  mask = (fpr <= fpr_thres)
  tpr_at_fpr = np.max(tpr * mask) # Maximum tpr at fpr <= fpr_thres
  roc_cutoff = np.sort(np.unique(mask*thres1))[1]
  pred = np.zeros_like(conf)
  pred[-conf>=roc_cutoff] = 1
  prec, rec, f1, _ = precision_recall_fscore_support(target, pred, average='binary')
  auc_value = auc(fpr, tpr)
  logger.log.info(f"TPR at FPR={fpr_thres} : {tpr_at_fpr:.3f}")
  logger.log.info(f"F1 : {f1:.3f}, prec: {prec:.3f}, recall: {rec:.3f}")
  logger.log.info(f"AUC: {auc_value:.3f}")
  if visualize:
    # ax = sns.boxplot(x='result_type', y='negative_conf', data=testset)
    # plt.show()
    fig, ax = plt.subplots()
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=50, density=False)
    x1 = testset.loc[testset.result_type==0, ['negative_conf']].values.squeeze()
    ax.hist(x=x1, label='clean', **kwargs)
    x2 = testset.loc[testset.result_type==1, ['negative_conf']].values.squeeze()
    ax.hist(x=x2, label='adv', **kwargs)
    ax.annotate(f'{int(roc_cutoff)}', xy=(roc_cutoff,0), xytext=(roc_cutoff,30), fontsize=14,
                arrowprops=dict(facecolor='black', width=1, shrink=0.1, headwidth=3))
    ax.legend()
    fig.savefig(os.path.join(logger.log_path, f"{mode}-hist.png"))

    # fig, ax = plt.subplots()
    # ax.plot(fpr, tpr)
    # ax.plot([0, 1], [0, 1], 'k--')
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    # fig.savefig(os.path.join(logger.log_path, f"{mode}-roc.png"))
    # plt.close()

  if log_metric:
    metrics = {"tpr":tpr_at_fpr, "fpr":fpr_thres, "f1":f1, "auc":auc_value}
    logger.log_metric(metrics)

  metric1 = (fpr, tpr, thres1)
  metric2 = (precision, recall, thres2)
  return metric1, metric2, tpr_at_fpr, f1, auc_value


def predict_attack(testset, confidence, conf_indices, fpr_thres=0.05, adv_ratio=None, visualize=False, by_class=False):
  # adv_count=None; visualize=True; fpr_thres=0.05
  target = np.array(testset['result_type'].tolist())
  target[target==-1] = 1
  conf = confidence.numpy()
  testset['negative_conf'] = -conf # negative of confidence : likelihood of adv. probability
  testset['log_likelihood'] = np.log((2 * np.pi) ** (-768//2)) * (0.5 * -conf)
  detection_result = np.zeros(len(target))

  if by_class:
    for idx in np.unique(conf_indices):
      per_cls_target = target[conf_indices==idx]
      per_cls_conf = conf[conf_indices==idx]
      fpr, tpr, thres1 = roc_curve(per_cls_target, -per_cls_conf)
      mask = (fpr > fpr_thres)
      # adv_count = sum(per_cls_target)
      if adv_ratio is None : #Choose threshold by fpr rate
        detect_thres = np.max(mask*thres1)
      else :
        adv_count = int(adv_ratio * len(per_cls_conf))
        detect_thres = np.sort(testset['negative_conf'].values[conf_indices==idx])[-adv_count] # adv_count th largest distance

      if visualize:
        ax = sns.boxplot(x='result_type', y='negative_conf', data=testset[(conf_indices == idx).numpy()])
        plt.show()
      detection_result[conf_indices == idx] = np.where(-per_cls_conf > detect_thres, 1, 0)
  else:
    # Class-agnostic
    fpr, tpr, thres1 = roc_curve(target, -conf)
    mask = (fpr > fpr_thres)
    if adv_ratio is None:  # Choose threshold by fpr rate
      detect_thres = np.max(mask * thres1)
    else:
      adv_count = int(adv_ratio * len(conf))
      # adv_count = sum(target)
      detect_thres = np.sort(testset['negative_conf'].values)[-adv_count]  # adv_count th largest distance

    if visualize:
      ax = sns.boxplot(x='result_type', y='negative_conf', data=testset)
      plt.show()
    detection_result[-conf > detect_thres] = 1

  testset['pred_type'] = detection_result.tolist()

def __get_test_features(model, tokenizer, batch_size, dataset, eps=1e-3):
  # dataset, batch_size, i = testset['text'].tolist(), 64, 0
  # gt = testset['ground_truth_output']
  num_samples = len(dataset)
  num_batches = int((num_samples // batch_size) + 1)
  features = []
  ce = nn.CrossEntropyLoss()

  # num_batches = 2
  for i in tqdm(range(num_batches)):
    lower = i * batch_size
    upper = min((i + 1) * batch_size, num_samples)
    examples = dataset[lower:upper]
    x = tokenizer(examples, padding='max_length', max_length=256,
                  truncation=True, return_tensors='pt')
    with torch.no_grad():
      output = model(input_ids=x['input_ids'].to(model.device), attention_mask=x['attention_mask'].to(model.device),
                     token_type_ids=(x['token_type_ids'].to(model.device) if 'token_type_ids' in x else None),
                     output_hidden_states=False)
      logits = output.logits
      _, pred = torch.max(logits, dim=1)
    model_embedding = model.bert.get_input_embeddings()
    x_emb = model_embedding(x['input_ids'].to(model.device))
    x_emb.retain_grad()
    output = model(inputs_embeds=x_emb.to(model.device), attention_mask=x['attention_mask'].to(model.device),
                   token_type_ids=(x['token_type_ids'].to(model.device) if 'token_type_ids' in x else None),
                   output_hidden_states=True, labels=pred)
    loss = output.loss
    loss.backward()

    x_calibrated = x_emb - eps * x_emb.grad
    with torch.no_grad():
      output = model(inputs_embeds=x_calibrated.to(model.device), attention_mask=x['attention_mask'].to(model.device),
                     token_type_ids=(x['token_type_ids'].to(model.device) if 'token_type_ids' in x else None),
                     output_hidden_states=True)
      feat = output.hidden_states[-1][:, 0, :].cpu()  # (Batch_size, 768)
      features.append(feat)

    # feat = output.hidden_states[-1]  # (Batch_size, 768)
    # pooled_feat = model.bert.pooler(feat)
    # features.append(pooled_feat.cpu())

  return torch.cat(features, dim=0)


def detect_attack_per_cls(testset, confidence, conf_indices, fpr_thres=0.05, visualize=False, by_class=False, logger=None, mode=None):
  """
  Detect attack for correct samples only to compute detection metric (TPR, recall, precision)
  """
  assert logger is not None, "Logger not given"
  # adv_count=None; visualize=True; fpr_thres=0.05
  target = np.array(testset['result_type'].tolist())
  target[target==-1] = 1
  conf = confidence.numpy()
  testset['negative_conf'] = -conf # negative of confidence : likelihood of adv. probability

  results = []
  # TODO : devise metric for whole class
  # tp_cnt = 0
  # fn_cnt = 0
  for idx in np.unique(conf_indices):
    per_cls_target = target[conf_indices==idx]
    per_cls_conf = conf[conf_indices==idx]
    fpr, tpr, thres1 = roc_curve(per_cls_target, -per_cls_conf)
    precision, recall, thres2 = precision_recall_curve(per_cls_target, -per_cls_conf)
    results.append([(fpr, tpr, thres1), (precision, recall, thres2)])
    mask = (fpr > fpr_thres)
    tpr_at_fpr = np.unique(tpr * mask)[1]  # Minimum tpr that is larger than 0
    logger.log.info(f"TPR at FPR={fpr_thres} : {tpr_at_fpr}")
    if visualize:
      # ax = sns.boxplot(x='result_type', y='negative_conf', data=testset[(conf_indices == idx).numpy()])
      # plt.show()
      ax = plt.subplot()
      kwargs = dict(histtype='stepfilled', alpha=0.3, bins=50, density=False)
      data = testset[(conf_indices==idx).numpy()]
      x1 = data.loc[data.result_type == 0, ['negative_conf']].values.squeeze()
      ax.hist(x=x1, label='clean', **kwargs)
      x2 = data.loc[data.result_type == 1, ['negative_conf']].values.squeeze()
      ax.hist(x=x2, label='adv', **kwargs)
      ax.set_title(f"Class {idx}")
      ax.legend()

  metric1 = (fpr, tpr, thres1)
  metric2 = (precision, recall, thres2)
  return metric1, metric2



def compute_ppl(texts):
  MODEL_ID = 'gpt2-large'
  print(f"Initializing {MODEL_ID}")
  model = GPT2LMHeadModel.from_pretrained(MODEL_ID).cuda()
  model.eval()
  tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID)
  encodings = tokenizer.batch_encode_plus(texts, add_special_tokens=True, truncation=True)

  batch_size = 1
  num_batch = len(texts) // batch_size
  eval_loss = 0
  likelihoods = []

  with torch.no_grad():
    for i in range(num_batch):
      start_idx = i * batch_size;
      end_idx = (i + 1) * batch_size
      x = encodings[start_idx:end_idx]
      ids = torch.LongTensor(x[0].ids)
      ids = ids.cuda()
      nll = model(input_ids=ids, labels=ids)[0] # negative log-likelihood
      likelihoods.append(-1 * nll.item())

  return torch.tensor(likelihoods)