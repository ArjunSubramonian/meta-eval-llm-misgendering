# This code in part draws from: https://github.com/tamannahossainkay/misgendered-backend.
# Please see the license here: https://github.com/tamannahossainkay/misgendered-backend?tab=readme-ov-file.
# Changes were made to the original code.

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

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import evaluate

import matplotlib.pyplot as plt
import pandas as pd
import re
import argparse
from pathlib import Path
import string
import time

import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

gender_codebook=pd.read_csv('pronouns.csv')

#Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--eval_file', type=Path, default='templates/eo_task.tsv') 
parser.add_argument('--output_path', type=Path, default='out/') 
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B") 
parser.add_argument('--num_generations', type=int, default=5) 
parser.add_argument('--device', type=str, default="cuda:5")
    
args = parser.parse_args()
eval_file=args.eval_file
output_path=args.output_path 

print('Model:', args.model)
print('Dataset: RUFF')
print()

#Load
MODEL=args.model
tokenizer = AutoTokenizer.from_pretrained(MODEL)

if '70b' in MODEL.lower() or '13b' in MODEL.lower() or '8x' in MODEL.lower():
    model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.to(args.device)
    
model.eval()

def label_to_cap(sentence):
  mask_start=sentence.find('{')
  if sentence[mask_start-2]=='.':
    return True
  else:
    False

def get_eval_instance(sentence, label, form):
  gender = gender_codebook['gender'][(gender_codebook == label).any(axis=1)].values[0]
  constraints=gender_codebook[form].tolist()

  if label_to_cap(sentence):
      label=label.capitalize()
      constraints=[c.capitalize() for c in constraints]
  return constraints, gender

all_pronouns = set(gender_codebook.iloc[:, 1:].values.flatten().tolist())

def misgendering_in_OLG(generations, acceptable_pronouns):
    num_wrong = 0
    breakdown = [1.0 for _ in range(len(generations))]
    
    for idx, generation in enumerate(generations):
        doc = nlp(generation)
        for token in doc:
            if token.text.lower() in all_pronouns:
                if token.text.lower() not in acceptable_pronouns:
                    breakdown[idx] = 0.0
                    num_wrong += 1
                break

    return 1 - num_wrong / len(generations), breakdown

def run_evaluation(eval_path):
    eval=pd.read_csv(eval_path, sep='\t')

    eval['base']=''
    eval['label']=''
    eval['loss_label']=0.0
    eval['pred']=''
    eval['loss_pred']=0.0
    eval['correct']=0
        
    #For each instance
    for idx, instance in eval.iterrows():
        if idx % 80 == 0:
            print(idx, '/', len(eval))
            
        #Get eval instance
        sentence=instance['sentence']
        form=instance['pronoun_type']
        label=instance['pronoun']
        
        constraints, gender=get_eval_instance(sentence, label, form)
        eval.at[idx, 'base']=gender
        eval.at[idx, 'label']=label
        
        #Evaluate
        losses=[]
        for c in constraints:
            choice=re.sub("\\" + form, c, sentence)

            input_ids=tokenizer(choice, return_tensors="pt")["input_ids"].to(args.device)
            out=model(input_ids, labels=input_ids)  
            loss=out['loss'].detach().item()
            losses.append(loss)

            if c == label:
                eval.at[idx, 'loss_label']=loss
      
        words=pd.DataFrame({'labels': constraints, 'losses':losses})
        words = words.sort_values('losses', ascending=True, ignore_index=True)
        
        eval.at[idx, 'pred']=words.labels[0]
        eval.at[idx, 'loss_pred']=words.losses[0]

        if words.labels[0]==label:
            eval.at[idx, 'correct']=1

    for g in range(args.num_generations):
        eval['greedy_pre_' + str(g + 1)] = ''
        eval['greedy_pre_' + str(g + 1) + '_correct'] = None
        eval['greedy_post_' + str(g + 1)] = ''
        eval['greedy_post_' + str(g + 1) + '_correct'] = None
    eval['greedy_pre_correct'] = None
    eval['greedy_post_correct'] = None
    
    for idx, instance in eval.iterrows():
        if idx % 80 == 0:
            print(idx, '/', len(eval))

        sentence=instance['sentence']
        form=instance['pronoun_type']
        label=instance['pronoun']
        
        choices=[instance['sentence'].split(form)[0], \
                 re.sub("\\" + form, label, sentence)]
        exps = ['pre', 'post']
        acceptable_pronouns = set(gender_codebook[(gender_codebook == label).any(axis=1)].iloc[0, 1:].tolist())
        
        for exp, choice in zip(exps, choices):
            contexts = [choice for _ in range(args.num_generations)]
            inputs = tokenizer(contexts, return_tensors="pt")
            prompt_length = inputs['input_ids'].shape[1]
            input_ids = inputs["input_ids"].to(args.device)
            attention_mask = inputs["attention_mask"].to(args.device)

            response = model.generate(input_ids, attention_mask=attention_mask, min_new_tokens=50, max_new_tokens=50, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, \
                                          num_beams=1, do_sample=True, top_k=50, top_p=0.95)
            generations = tokenizer.batch_decode(response[:, prompt_length:], skip_special_tokens=True)
            for g in range(args.num_generations):
                eval.at[idx, f'greedy_{exp}_' + str(g + 1)] = choice + generations[g]
            eval.at[idx, f'greedy_{exp}_correct'], breakdown = misgendering_in_OLG(generations, acceptable_pronouns)
            for g in range(args.num_generations):
                eval.at[idx, f'greedy_{exp}_' + str(g + 1) + '_correct'] = breakdown[g]
    return eval

directory = os.path.join(output_path,args.model.split('/')[-1])

if not os.path.exists(directory):
    os.makedirs(directory)

#Evaluate
eval=run_evaluation(eval_file)

#Save Raw results
for pronoun in gender_codebook['gender']:
    file_name=pronoun + '.csv'
    file_path=os.path.join(output_path,args.model.split('/')[-1],file_name)
    sub_eval = eval[eval['base'] == pronoun].reset_index()
    sub_eval.to_csv(file_path)
    sub_eval = pd.read_csv(file_path)