# This code in part draws from: https://peteris.rocks/blog/measuring-the-repetitiveness-of-a-text/,
# with some adaptations and corrections.

import os
import random
import numpy as np
import torch

# Set random seed
seed_value = 42

os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import pandas as pd
import matplotlib.pyplot as plt 
import spacy
nlp = spacy.load('en_core_web_sm')

from collections import defaultdict

def ngrams(tokens, n):
  return zip(*[tokens[i:] for i in range(n)])

def count_ngrams(tokens, n):
  counts = defaultdict(int)
  for ngram in ngrams(tokens, n):
    counts[ngram] += 1
  return counts

def rr(tokens, max_n=4):
  if len(tokens) < max_n:
    raise Exception('Too few tokens, change max_n')

  result = 1.0

  for n in range(1, max_n+1):
    ngram_counts = count_ngrams(tokens, n)
    singletons = [ngram for ngram, count in ngram_counts.items() if count == 1]
    numerator = len(ngram_counts) - len(singletons)
    denominator = len(ngram_counts)
    result *= numerator / denominator

  return pow(result, 1.0/max_n)

cols = []
for g in range(1, 6):
    cols.append(f'olg_{str(g)}')

model_names = ['Llama-3.1-8B', 'Llama-3.1-70B', \
               'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1-4bit', \
               'OLMo-2-1124-7B', 'OLMo-2-1124-13B']

data = []
data_cols = ["he", "she", "they", "xe"]

exclude_models = set([])

for model in sorted(model_names):
    if model in exclude_models:
        continue
    
    data.append([model] + [None] * len(data_cols))
    
    files = ["he.csv", "she.csv", "they.csv", "xe.csv"]

    rr_scores_list = []
    for it, f in enumerate(data_cols):
        f += '.csv'
        
        df = pd.read_csv("./out/" + model + "/" + f)
        print(model, f.split('.')[0])

        rr_scores = []
        texts = []
        for t in df[cols].values.reshape(-1):
            doc = nlp(t)
            tokens = [token.text.lower() for token in doc]
            
            rr_score = rr(tokens)
            rr_scores.append(rr_score)
            texts.append(t)

        res = f'{np.mean(np.array(rr_scores)):.3f}' + ' ± ' + f'{np.std(np.array(rr_scores)):.3f}'
        print('Repetitiveness (mu ± sig):', res)

        sorted_rr_scores = sorted(zip(rr_scores, texts), reverse=True)
        for s, t in sorted_rr_scores[:1]:
            print(s)
            print(t)
            print()

        rr_scores_list.append(rr_scores)
        data[-1][it + 1] = res

    # plt.figure()
    # plt.boxplot(rr_scores_list)
    # plt.show()

df = pd.DataFrame(data, columns=[''] + data_cols)
print(df.to_latex(na_rep='', index=False))