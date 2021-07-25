import os
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import transformers
import torch
import torch.nn as nn

device = torch.device('cuda')
# model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-rotten-tomatoes")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
embedding = model.bert.embeddings.word_embeddings
weight = embedding.weight.detach()
cos_sim = nn.CosineSimilarity(dim=0)


normalized = weight / torch.norm(weight, p=2, dim=-1).unsqueeze(-1)
gram_mtx = torch.matmul(normalized, normalized.T)
print(gram_mtx[0])
print(cos_sim(weight[3185], weight[2143]))
# cnts = gram_mtx.gt(0.9).sum(dim=1)
# index = torch.where(cnts>1, torch.arange(len(cnts)), -1)
# index = index[index>-1]
# tokenizer.batch_decode(index[995:])



from textattack.shared import AbstractWordEmbedding, WordEmbedding
emb = WordEmbedding.counterfitted_GLOVE_embedding()
weight = torch.tensor(emb.embedding_matrix)
normalized = weight / torch.norm(weight, p=2, dim=-1).unsqueeze(-1)
gram_mtx = torch.matmul(normalized, normalized.T)

output = []
unit = 1000
rounds = gram_mtx.shape[0] // unit +1
for r in range(rounds):
  cnt = gram_mtx[r*unit : (r+1)*unit].gt(0.9).sum(dim=1).tolist()
  output.extend(cnt)

from transformers import GPT2PreTrainedModel, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
device = 'cuda'
model = model.to(device)


from textattack.attack_results.attack_result import AttackResult
import pandas as pd
from termcolor import colored

def pretty_print(text, color):
  words = text.split(' ')
  for w in words :
    if w[0] == '[' and w[1] == '[':
      print(colored(w, color), end=' ')
    else:
      print(w, end=' ')

result = pd.read_csv('bert-base-uncased-imdb.csv')
successed = result[result.result_type=='Successful']
for idx in range(len(successed)):
  s = successed.iloc[idx]
  print(f"{idx}")
  pretty_print(s.original_text, 'red')
  print('\n')
  pretty_print(s.perturbed_text, 'green')
  print('\n')


from textattack.constraints.grammaticality import PartOfSpeech, LanguageTool
lt = LanguageTool(0)

adv_text = "Now I like Victor Herbert. And I like Mary Martin and Allan Jones. But it would have been nice to see a real biography of Victor Herbert. Walter Connolly as Herbert does have a decent resemblance to him in his latter years<br /><br />Jones and Martin [[sing]] beautifully though. The Herbert music is just there to adorn the plot line concerning these two musical [[performers]]. Jones's John Ramsay is a frail character, very similar to Gaylord Ravenal in Showboat who Jones also played.<br /><br />As for Mary Martin, it's a mystery why she never had a good Hollywood [[career]]. She did films with Bing Crosby and Dick Powell as well as this one. She performed well, but movie audiences didn't take to her. The best musical moment in the film is Jones and Martin in a duet of Thine Alone. The recordings I have of the song are individual and it was written as a duet. There's also a pleasant scene with Jones and Martin riding bicycles swapping Herbert songs as they ride.<br /><br />The [[real]] Victor Herbert with his womanizing and his Irish patriot background and his musical training in Germany where he developed a love for all things German would have been a fascinating study. He was also a cello virtuoso before he turned full time to composing. I have to take strong exception to the reviewer who said Cuddles Sakall would have been a good Victor Herbert. Sakall as Irish, HELLO.<br /><br />Nice movie, but the real Vic would have been so much better. "
output = lt.lang_tool.check(adv_text)