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

from datasets import load_dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import evaluate
from neutralize import convert
from deneutralize import revert

import pandas as pd
import re
import argparse
from pathlib import Path
import string
import matplotlib.pyplot as plt
import time

import spacy
from spacy import displacy
from spacy.symbols import punct, nsubj, VERB
nlp = spacy.load('en_core_web_sm')

dataset = load_dataset("alexaAI/TANGO", data_files={'misgendering': 'misgendering.jsonl'})['misgendering'].to_pandas()
pronouns = pd.read_csv('tango-pronouns.csv')

#Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pronoun', type=str, default='xe') 
parser.add_argument('--output_path', type=Path, default='out/') 
parser.add_argument('--model', type=str, default="allenai/OLMo-2-1124-7B") 
parser.add_argument('--num_generations', type=int, default=5) 
parser.add_argument('--device', type=str, default="cuda:5")
    
args = parser.parse_args()
output_path=args.output_path 
label = args.pronoun

print('Model:', args.model)
print('Dataset: TANGO')
print('Pronoun:', label)
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


def declare_pronouns(gender, sentence):
    row=pronouns[pronouns['gender']==gender]
    row=row.reset_index()
    
    sentence=re.sub('{nom}', row['nom'][0],sentence)
    sentence=re.sub('{acc}', row['acc'][0],sentence)
    sentence=re.sub('{gen_att}', row['gen_att'][0],sentence)
    sentence=re.sub('{gen_pred}', row['gen_pred'][0], sentence)
    sentence=re.sub('{ref}', row['ref'][0],sentence)

    sentence=re.sub('{Nom}', row['nom'][0].capitalize(),sentence)
    sentence=re.sub('{Acc}', row['acc'][0].capitalize(),sentence)
    sentence=re.sub('{Gen_att}', row['gen_att'][0].capitalize(),sentence)
    sentence=re.sub('{Gen_pred}', row['gen_pred'][0].capitalize(), sentence)
    sentence=re.sub('{Ref}', row['ref'][0].capitalize(),sentence)

    sentence=re.sub('{NOM}', row['nom'][0].upper(),sentence)
    sentence=re.sub('{ACC}', row['acc'][0].upper(),sentence)
    sentence=re.sub('{GEN_ATT}', row['gen_att'][0].upper(),sentence)
    sentence=re.sub('{GEN_PRED}', row['gen_pred'][0].upper(), sentence)
    sentence=re.sub('{REF}', row['ref'][0].upper(),sentence)
    
    return sentence

all_pronouns = set(pronouns.iloc[:, 1:].values.flatten().tolist())

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

def run_evaluation():
    eval = dataset[dataset['pronoun_family'] == label]
    eval = eval.reset_index()
    print('Total instances:', len(eval))
    
    for g in range(args.num_generations):
        eval['olg_' + str(g + 1)]=None
        eval['olg_' + str(g + 1) + '_correct']=None
        eval['olg_template_' + str(g + 1)]=None
        eval['olg_template_' + str(g + 1) + '_correct']=None
        
    eval['gen_correct']=None
    eval['prob_correct']=None
    eval['total_templates']=None

    for idx, instance in eval.iterrows():

        if idx % 20 == 0:
            print(idx, '/', len(eval))
        
        eval.at[idx, 'template'] = eval.at[idx, 'template'].capitalize()
        
        generations = []
        templates = [eval.at[idx, 'template'] for _ in range(args.num_generations)]
        inputs = tokenizer(templates, return_tensors="pt")
        prompt_length = inputs['input_ids'].shape[1]
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)

        response = model.generate(input_ids, attention_mask=attention_mask, min_new_tokens=50, max_new_tokens=50, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, \
                                      num_beams=1, do_sample=True, top_k=50, top_p=0.95)
        generations = tokenizer.batch_decode(response[:, prompt_length:], skip_special_tokens=True)
        for g in range(args.num_generations):
            eval.at[idx, 'olg_' + str(g + 1)] = eval.at[idx, 'template'] + generations[g]
    
        acceptable_pronouns = set(pronouns[(pronouns == label.lower()).any(axis=1)].iloc[0, 1:].tolist())
        eval.at[idx, 'gen_correct'], breakdown = misgendering_in_OLG(generations, acceptable_pronouns)
        for g in range(args.num_generations):
            eval.at[idx, 'olg_' + str(g + 1) + '_correct'] = breakdown[g]

        accs = []
        for g in range(args.num_generations):
            sent_piece = generations[g]
            orig_doc = nlp(sent_piece)
            reconstructed_sent = ''
            pronoun_count = 0
            for token in orig_doc:
                if (pronouns == token.text.lower()).any().any():
                    if pronoun_count > 0:
                        break
                    else:
                        pronoun_count += 1
                
                # Convert "xe" to "she"
                if (pronouns.iloc[3:, :] == token.text.lower()).any().any():
                    mask = (pronouns.iloc[:, 2:] == token.text.lower()).any(axis=0)
                    cols = pronouns.columns[2:]
                    
                    if token.text.istitle():
                        reconstructed_sent += pronouns.iloc[1][cols[mask][0]].capitalize() + token.whitespace_
                    elif token.text.isupper():
                        reconstructed_sent += pronouns.iloc[1][cols[mask][0]].upper() + token.whitespace_
                    else:
                        reconstructed_sent += pronouns.iloc[1][cols[mask][0]] + token.whitespace_
                else:
                    reconstructed_sent += token.text + token.whitespace_
            
            neutralized_sent = convert(reconstructed_sent)
            template = revert(neutralized_sent)
            eval.at[idx, 'olg_template_' + str(g + 1)] = eval.at[idx, 'template'] + template

            for p in pronouns.iloc[2, :].tolist():
                if p in template.lower():
                    print('Possible failure to catch pronoun!')
                    print(p, template)
                    print()

            if eval.at[idx, 'olg_template_' + str(g + 1)].count('{') > 1:
                print('Possibly multiple [MASK] tokens!')
                print(eval.at[idx, 'olg_template_' + str(g + 1)])
                print()
            
            if declare_pronouns('he', eval.at[idx, 'olg_template_' + str(g + 1)]) == eval.at[idx, 'olg_template_' + str(g + 1)]:
                eval.at[idx, 'olg_template_' + str(g + 1)] = None
                print('Template not successfully created!')
                continue
    
            choices = [('they', (eval.at[idx, 'template'] + neutralized_sent).strip())]
            for option in pronouns['gender'].tolist():
                if option == 'they':
                    continue
                choices.append((option, \
                               declare_pronouns(option, eval.at[idx, 'olg_template_' + str(g + 1)]).strip()))
    
            losses = []
            for pronoun, choice in choices:
                input_ids=tokenizer(choice, return_tensors="pt")["input_ids"].to(args.device)
                out=model(input_ids, labels=input_ids)  
                loss=out['loss'].detach().item()
                losses.append((pronoun, loss))
            
            pred = min(losses, key=lambda tup: tup[1])[0]
            eval.at[idx, 'olg_template_' + str(g + 1) + '_correct']=int(pred == label)
            accs.append(int(pred == label))

        if len(accs) > 0:
            eval.at[idx, 'prob_correct'] = sum(accs) / len(accs)
        eval.at[idx, 'total_templates'] = len(accs)
    
    return eval

file_name=label + '.csv'
file_path=os.path.join(output_path,args.model.split('/')[-1],file_name)
directory = os.path.join(output_path,args.model.split('/')[-1])

if not os.path.exists(directory):
    os.makedirs(directory)

#Evaluate
eval=run_evaluation()

#Save Raw results
eval.to_csv(file_path)

eval = pd.read_csv(file_path)