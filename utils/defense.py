from tqdm import tqdm
import torch
import numpy as np

def predict_decision_boundary(model, batch_size, adv_features, conf_indices, train_stats, gt):
  #For loop per adv_features:
    # Exclude closest cls.
    # Build intermediate features connecting the mean and the features per class
    # forward with BertPooler -> classifier
    # Find minimum point at which the prediction changes to the corresponding cls.
  pass

def refine_predict_by_topk(model, tokenizer, top_k, use_ratio, batch_size, dataset, gt):
  num_samples = len(dataset)
  num_batches = int((num_samples // batch_size) + 1)
  features = []
  correct = 0
  total = 0
  refined_correct = 0

  with torch.no_grad():
    for i in tqdm(range(num_batches)):
      lower = i * batch_size
      upper = min((i + 1) * batch_size, num_samples)
      examples = dataset[lower:upper]
      y = torch.LongTensor(gt[lower:upper])
      x = tokenizer(examples, padding='max_length', max_length=256,
                    truncation=True, return_tensors='pt')
      output = model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                     token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None),
                     output_hidden_states=True, output_attentions=True)

      _, pred = torch.max(output.logits, dim=1)
      cls_token_attention = output.attentions[-1][:,:,0].cpu() # (attentions shape : (batch_size, num_heads, seq_len, seq_len)
      cls_token_attention *= x['attention_mask'].unsqueeze(1) # Mask out attention on pad tokens just in case
      cls_token_attention = cls_token_attention.sum(1) # Summation on head dimension
      attention_sorted, top_k_token_idx = cls_token_attention.sort()

      top_k_percent = top_k
      top_ks = [int(example.nonzero().numel()*top_k_percent) for example in x['input_ids']]
      max_top_k = max(top_ks)
      max_length = attention_sorted.size(1)
      seq_len = [example.nonzero().numel() for example in x['input_ids']]

      arg_replace_idx = torch.empty((1,max_top_k), dtype=torch.long)
      if use_ratio :
        col_idx = top_k_token_idx[:, - max_top_k:]
        for row, k in enumerate(top_ks):
          mask_len = max_top_k - k
          col_idx[row, list(range(mask_len))] = max_length-1  # Mask out by giving the last token index
        row_idx = torch.arange(cls_token_attention.size(0)).repeat(max_top_k, 1).permute(1, 0).reshape(1,-1).squeeze().long()
      else :
        col_idx = top_k_token_idx[:, - top_k_percent:]
        row_idx = torch.arange(cls_token_attention.size(0)).repeat(top_k_percent,1).permute(1,0).reshape(1,-1).squeeze().long()
      col_idx = col_idx.reshape(-1,1).squeeze().long()
      pad_token = 103 # [MASK]
      refined_x = x['input_ids'].index_put(indices=(row_idx, col_idx), values=torch.tensor(pad_token, dtype=torch.long))

      output = model(input_ids=refined_x.cuda(), attention_mask=x['attention_mask'].cuda(),
                     token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None),
                     output_hidden_states=True, output_attentions=True)

      _, refined_pred = torch.max(output.logits, dim=1)
      correct += (y.eq(pred.cpu())).sum().item()
      refined_correct += (y.eq(refined_pred.cpu())).sum().item()
      total += len(examples)
      # feat = output.hidden_states[-1]  # (Batch_size, 768)
      # pooled_feat = model.bert.pooler(feat)
      # features.append(pooled_feat.cpu())
  print(f"Original Adv. Acc. {correct/total:.3f} --> {refined_correct/total:.3f}")
  return refined_correct, total

def refine_predict_by_prob(model, tokenizer, batch_size, dataset, gt, prob):
  # dataset, batch_size, i, gt = testset.loc[testset['result_type']==1,'text'].tolist(), 256, 0, testset.loc[testset['result_type']==1,'ground_truth_output'].values
  num_samples = len(dataset)
  num_batches = int((num_samples // batch_size) + 1)
  features = []
  correct = 0
  total = 0
  refined_correct = 0

  with torch.no_grad():
    for i in tqdm(range(num_batches)):
      lower = i * batch_size
      upper = min((i + 1) * batch_size, num_samples)
      examples = dataset[lower:upper]
      y = torch.LongTensor(gt[lower:upper])
      x = tokenizer(examples, padding='max_length', max_length=256,
                    truncation=True, return_tensors='pt')
      output = model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                     token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None),
                     output_hidden_states=True, output_attentions=True)

      _, pred = torch.max(output.logits, dim=1)
      cls_token_attention = output.attentions[-1][:,:,0].cpu() # (attentions shape : (batch_size, num_heads, seq_len, seq_len)
      cls_token_attention *= x['attention_mask'].unsqueeze(1) # Mask out attention on pad tokens just in case ; Shape (batch_size, num_heads, seq_len)
      # cls_token_attention = cls_token_attention.sum(1) # Summation on head dimension
      row_idx, _, col_idx = (cls_token_attention > prob).nonzero(as_tuple=True)
      pad_token = 103 # [MASK]
      refined_x = x['input_ids'].index_put(indices=(row_idx, col_idx), values=torch.tensor(pad_token, dtype=torch.long))

      output = model(input_ids=refined_x.cuda(), attention_mask=x['attention_mask'].cuda(),
                     token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None),
                     output_hidden_states=True, output_attentions=True)

      _, refined_pred = torch.max(output.logits, dim=1)
      correct += (y.eq(pred.cpu())).sum().item()
      refined_correct += (y.eq(refined_pred.cpu())).sum().item()
      total += len(examples)
      # feat = output.hidden_states[-1]  # (Batch_size, 768)
      # pooled_feat = model.bert.pooler(feat)
      # features.append(pooled_feat.cpu())
  print(f"Original Acc. {correct/total}")
  print(f"Refined Acc. {refined_correct/total}")
  return refined_correct, total

def predict_by_confidence(confidence, distance, gt, adv_count, total):
  # (Oracle Detection) Accuracy
  # perturbed_ind = testset['result_type'].values
  # adv_correct = adv_count - conf_indices[torch.tensor(perturbed_ind==1)].eq(torch.tensor(y[perturbed_ind==1])).sum()
  # clean_correct = conf_indices[torch.tensor(perturbed_ind==0)].eq(torch.tensor(y[perturbed_ind==0])).sum()
  # print(f"Adv. Acc. {adv_correct/adv_count}")
  # print(f"Clean Acc. {clean_correct/(total-adv_count)}")
  # print(f"Total Acc. {(adv_correct+clean_correct) / total}")
  # Accuracy using threshold
  thres = confidence.sort().values[adv_count]
  indices_sorted = torch.sort(distance, dim=1)[1]  # Get sorted indexes (not values)
  conf_indices = indices_sorted[:, -2]  # Second largest index
  # conf_indices = torch.min(distance, dim=1)[1]
  adv_correct = conf_indices[confidence < thres].eq(torch.tensor(gt[confidence < thres])).sum()
  conf_indices = torch.max(distance, dim=1)[1]
  clean_correct = conf_indices[confidence > thres].eq(torch.tensor(gt[confidence > thres])).sum()
  print(f"Adv. Acc. {adv_correct / adv_count}")
  print(f"Clean Acc. {clean_correct / (total - adv_count)}")
  print(f"Total Acc. {(adv_correct + clean_correct) / total}")

def predict_clean_samples(model, tokenizer, batch_size, dataset, gt):
  # Compute Acc. on dataset
  num_samples = len(dataset)
  num_batches = int((num_samples // batch_size) + 1)

  correct = 0
  total = 0
  # num_batches = 2
  with torch.no_grad():
    for i in tqdm(range(num_batches)):
      lower = i * batch_size
      upper = min((i + 1) * batch_size, num_samples)
      examples = dataset[lower:upper]
      labels = gt[lower:upper]
      y = torch.LongTensor(labels).cuda()
      x = tokenizer(examples, padding='max_length', max_length=256,
                    truncation=True, return_tensors='pt')
      output = model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                     token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None),
                     output_hidden_states=True)
      preds = torch.max(output.logits, dim=1).indices
      correct += y.eq(preds).sum().item()
      total += preds.size(0)


  print(f"Clean Acc. {correct/total:.3f}")
  return correct, total

def predict_dataset(model, tokenizer, batch_size, dataset, text_key):
  # Compute Acc. on dataset
  num_samples = len(dataset['label'])
  label_list = np.unique(dataset['label'])
  num_labels = len(label_list)
  num_batches = int((num_samples // batch_size) + 1)

  correct = 0
  total = 0
  # num_batches = 2
  with torch.no_grad():
    for i in tqdm(range(num_batches)):
      lower = i * batch_size
      upper = min((i + 1) * batch_size, num_samples)
      examples = dataset[text_key][lower:upper]
      labels = dataset['label'][lower:upper]
      y = torch.LongTensor(labels).cuda()
      x = tokenizer(examples, padding='max_length', max_length=256,
                    truncation=True, return_tensors='pt')
      output = model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                     token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None),
                     output_hidden_states=True)
      preds = torch.max(output.logits, dim=1).indices
      correct += y.eq(preds).sum()
      total += preds.size()
      feat = output.hidden_states[-1][:, 0, :].cpu()  # (Batch_size, 768)