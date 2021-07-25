import math
import os
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve


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


def get_train_features(model, tokenizer, args, batch_size, dataset, text_key, layer=-1):
  assert layer=='pooled' or layer < 0 , "Layer either has to be a int between -12~-1 or the pooling layer"
  model_name = os.path.basename(args.target_model)
  model_name += f"-layer_{layer}"

  if os.path.exists(f"saved_feats/{model_name}.pkl"):
    with open(f"saved_feats/{model_name}.pkl", "rb") as handle:
      features = pickle.load(handle)
    return features
  print("Building train features")
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
      x = tokenizer(examples, padding='max_length', max_length=256,
                    truncation=True, return_tensors='pt')
      output = model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                     token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None),
                     output_hidden_states=True)
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

  with open(f"saved_feats/{model_name}.pkl", "wb") as f:
    pickle.dump(features, f)
  return features


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
      output = model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                     token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None),
                     output_hidden_states=False)
      logits = output.logits
      _, pred = torch.max(logits, dim=1)
    model_embedding = model.bert.get_input_embeddings()
    x_emb = model_embedding(x['input_ids'].cuda())
    x_emb.retain_grad()
    output = model(inputs_embeds=x_emb.cuda(), attention_mask=x['attention_mask'].cuda(),
                   token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None),
                   output_hidden_states=True, labels=pred)
    loss = output.loss
    loss.backward()

    x_calibrated = x_emb - eps * x_emb.grad
    with torch.no_grad():
      output = model(inputs_embeds=x_calibrated.cuda(), attention_mask=x['attention_mask'].cuda(),
                     token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None),
                     output_hidden_states=True)
      feat = output.hidden_states[-1][:, 0, :].cpu()  # (Batch_size, 768)
      features.append(feat)

    # feat = output.hidden_states[-1]  # (Batch_size, 768)
    # pooled_feat = model.bert.pooler(feat)
    # features.append(pooled_feat.cpu())

  return torch.cat(features, dim=0)

def get_test_features(model, tokenizer, batch_size, dataset, perturb=False, topk=5, layer=-1, use_cosine_sim=True):
  # dataset, batch_size, i, layer = testset['text'].tolist(), 32, 0, -1
  num_samples = len(dataset)
  num_batches = int((num_samples // batch_size) + 1)
  features = []
  probs = []

  # num_batches = 2
  with torch.no_grad():
    for i in tqdm(range(num_batches)):
      lower = i * batch_size
      upper = min((i + 1) * batch_size, num_samples)
      examples = dataset[lower:upper]
      x = tokenizer(examples, padding='max_length', max_length=256,
                    truncation=True, return_tensors='pt')
      output = model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                     token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None),
                     output_hidden_states=True, output_attentions=True)

      if type(layer) == int :
        feat = output.hidden_states[layer][:, 0, :].cpu()  # output.hidden_states : (Batch_size, sequence_length, hidden_dim)
        if use_cosine_sim: # 10~20 seems fine
          query = output.hidden_states[layer][:, 0, :].cuda() #(batch_size, hidden_dim)
          word_vectors = output.hidden_states[layer-1] #(batch_size, seq  , hidden_dim)
          # token_mask = x['attention_mask'][:,1:,None].cuda()
          # word_vectors = torch.gather(word_vectors, token_mask)
          word_vectors = word_vectors / torch.norm(word_vectors, p=2, dim=-1, keepdim=True)
          word_vectors = word_vectors * x['attention_mask'][:,:,None].cuda()
          query = query / torch.norm(query, p=2, dim=-1, keepdim=True)
          inner_product = torch.einsum("bh, bsh -> bs", query, word_vectors)
          if False :
            inner_product[inner_product <= 0] = 1e-12
            inner_product = inner_product / torch.sum(inner_product, dim=-1, keepdim=True)
          else:
            T = 0.5
            inner_product = torch.exp(inner_product/T) / torch.exp(inner_product/T).sum(-1, keepdims=True)
          topk_log_prob = torch.log(torch.topk(inner_product, topk).values)
          log_prob_sum = torch.sum(topk_log_prob,1, keepdim=True)
          features.append(feat.cpu())
          probs.append(log_prob_sum.cpu())

        else : #50-150 seems fine for use_key, 5~10 for use_query
          use_key=True
          if use_key:
            conditional_prob = output.attentions[layer][:,:,:,0].sum(1).cpu() #attention score using cls token as key. Then, sum across heads
          else :
            conditional_prob = output.attentions[layer][:,:,0,:].sum(1).cpu() #attention score using cls token as query. Then, sum across heads

          conditional_prob = conditional_prob / conditional_prob.sum(-1, keepdim=True) #(batch_size, seq_len)
          # conditional_prob[conditional_prob <= 0] = 1
          conditional_prob = conditional_prob.double()
          topk_prob = torch.topk(conditional_prob, topk).values
          topk_prob[topk_prob <=0] = 1
          log_prob_sum = torch.sum(torch.log(topk_prob), dim=-1)
          # product_prob = torch.sum(topk_log_prob,1, keepdim=True)
          features.append(feat)
          probs.append(log_prob_sum.cpu())

      else:
        feat = output.hidden_states[-1]  # (Batch_size, 768)
        pooled_feat = model.bert.pooler(feat)
        features.append(pooled_feat.cpu())
        probs.append(torch.empty(pooled_feat.shape[0],1))

  return torch.cat(features, dim=0), torch.cat(probs, dim=0)

import math
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
  confidence, conf_indices = torch.max(output, dim=1) # Takes the min of class conditional probability
  if use_marginal:
    confidence = torch.log(torch.sum(torch.exp(output), dim=1)) # Takes the marginal probability
  return confidence, conf_indices, output


def detect_attack(testset, confidence, conf_indices, fpr_thres=0.05, visualize=False, by_class=False):
  """
  Detect attack for correct samples only to compute detection metric (TPR, recall, precision)
  """
  # adv_count=None; visualize=True; fpr_thres=0.05
  target = np.array(testset['result_type'].tolist())
  target[target==-1] = 1
  conf = confidence.numpy()
  testset['negative_conf'] = -conf # negative of confidence : likelihood of adv. probability


  if by_class:
    # TODO : devise metric for whole class
    # tp_cnt = 0
    # fn_cnt = 0
    for idx in np.unique(conf_indices):
      per_cls_target = target[conf_indices==idx]
      per_cls_conf = conf[conf_indices==idx]
      fpr, tpr, thres1 = roc_curve(per_cls_target, -per_cls_conf)
      precision, recall, thres2 = precision_recall_curve(per_cls_target, -per_cls_conf)
      mask = (fpr > fpr_thres)
      tpr_at_fpr = np.unique(tpr * mask)[1]  # Minimum tpr that is larger than 0
      print(f"TPR at FPR={fpr_thres} : {tpr_at_fpr}")
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
        plt.show()

  else:
    # Class-agnostic
    fpr, tpr, thres1 = roc_curve(target, -conf)
    precision, recall, thres2 = precision_recall_curve(target, -conf)
    mask = (fpr > fpr_thres)
    tpr_at_fpr = np.unique(tpr * mask)[1]  # Minimum tpr that is larger than 0
    print(f"TPR at FPR={fpr_thres} : {tpr_at_fpr}")
    if visualize:
      # ax = sns.boxplot(x='result_type', y='negative_conf', data=testset)
      # plt.show()
      ax = plt.subplot()
      kwargs = dict(histtype='stepfilled', alpha=0.3, bins=50, density=False)
      x1 = testset.loc[testset.result_type==0, ['negative_conf']].values.squeeze()
      ax.hist(x=x1, label='clean', **kwargs)
      x2 = testset.loc[testset.result_type==1, ['negative_conf']].values.squeeze()
      ax.hist(x=x2, label='adv', **kwargs)
      ax.legend()
      plt.show()

  metric1 = (fpr, tpr, thres1)
  metric2 = (precision, recall, thres2)
  return metric1, metric2


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

