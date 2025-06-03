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

model_names = ['Llama-3.1-8B', 'Llama-3.1-70B', \
               'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1-4bit',
               'OLMo-2-1124-7B', 'OLMo-2-1124-13B']
files = ["he.csv", "she.csv", "they.csv", "xe.csv"]

if not os.path.exists('./samples/'):
    os.makedirs('./samples/')

for model in model_names:

    if not os.path.exists('./samples/' + model):
        os.makedirs('./samples/' + model)

    for f in files:
    
        for exp in ['pre', 'post']:
            df = pd.read_csv("./out/" + model + "/" + f)
            
            cols = []
            for i in range(5):
                cols.append(['sentence', f'greedy_{exp}_' + str(i + 1), f'greedy_{exp}_' + str(i + 1) + '_correct'])

            samples = []
            for col in cols:
                samples.append(df[col].sample(n=5, random_state=131719).values)
        
            all_samples = np.concatenate(samples)
            df_subset = pd.DataFrame({
                'template': all_samples[:, 0],
                f'random_{exp}_gen': all_samples[:, 1],
                'correct': all_samples[:, 2],
            })
            df_subset.to_csv(f"samples/{model}/{f.split('.')[0]}_{exp}_mask_gen.csv")