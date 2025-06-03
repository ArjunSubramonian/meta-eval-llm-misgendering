import pandas as pd
import numpy as np
import random

from scipy.stats import sem

import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from matplotlib.transforms import Affine2D

ids = [374562381, 1341630721, 988571976, 1655461561, 2328415, 1495048709, \
       1126667510, 1723069802, 515422913, 50540968, 538891487, 1197190271]

model_name_map = {
    'Llama-3.1-70B': 'Llama-70B', \
    'Llama-3.1-8B': 'Llama-8B', \
    'Mixtral-8x22B-v0.1-4bit': 'Mixtral-8x22B', \
    'Mixtral-8x7B-v0.1': 'Mixtral-8x7B', \
    'OLMo-2-1124-13B': 'OLMo-13B', \
    'OLMo-2-1124-7B': 'OLMo-7B'
}

model_names = ['Llama-3.1-8B', 'Llama-3.1-70B', \
               'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1-4bit', \
               'OLMo-2-1124-7B', 'OLMo-2-1124-13B']

annot_map = {
    "correct": 1,
    "misgendering": 0,
    "no_pronoun": 1,
    "plural_they": 1
}
annots = sorted(list(annot_map.keys()))

data_cols = ["he", "she", "they", "xe"]

agr_mean_data = {
    'pre': [],
    'post': []
}
agr_std_data = {
    'pre': [],
    'post': []
}
agr_samples = {
    'pre': [],
    'post': []
}

extr_mean_data = {
    'pre': [],
    'post': []
}
extr_std_data = {
    'pre': [],
    'post': []
}
extr_samples = {
    'pre': [],
    'post': []
}

annot_count_data = {
    'pre': [],
    'post': []
}

for exp in ['pre', 'post']:
    for gid in ids:
        df = pd.read_csv('https://docs.google.com/spreadsheets/d/' + 
                           '14a9qqtU86_AFOwqaqy63O-SJKr_fXAOesezt-DDvs10' +
                           f'/export?gid={gid}&format=csv',
                           # Set first column as rownames in data frame
                           index_col=0,
                          )
        gen_col = f'random_{exp}_gen'
        if gen_col not in df.columns:
            continue

        annot = 'vagrant'
        if f'{annot}_label_pronoun' not in df.columns:
            annot = 'preethi'

        model = df['model'].iloc[0]

        conv_model_name = model_name_map[model]
        
        agr_mean_data[exp].append([conv_model_name] + [None] * len(data_cols))
        agr_std_data[exp].append([conv_model_name] + [None] * len(data_cols))
        agr_samples[exp].append([conv_model_name] + [[]])
        
        extr_mean_data[exp].append([conv_model_name] + [None] * len(data_cols))
        extr_std_data[exp].append([conv_model_name] + [None] * len(data_cols))
        extr_samples[exp].append([conv_model_name] + [[]])
        
        annot_count_data[exp].append([conv_model_name] + [None] * len(data_cols))

        for it, pro in enumerate(data_cols):
            
            sub_df = df[df["pronoun"] == pro]

            human_ratings = sub_df[f'{annot}_label_pronoun'].apply(lambda x: annot_map[x]).values
            gen_ratings = sub_df['correct'].values

            agr_mean = np.mean(human_ratings == gen_ratings)
            agr_std = sem(human_ratings == gen_ratings)

            examples = sub_df[gen_col][human_ratings != gen_ratings]
            n = min(len(examples), 1)
            examples = examples.sample(n=n, random_state=131719).values
            if n > 0:
                agr_samples[exp][-1][1].append(examples[0])

            print(exp, annot, model, pro, agr_mean, agr_std)

            agr_mean_data[exp][-1][it + 1] = agr_mean
            agr_std_data[exp][-1][it + 1] = agr_std

            extr_ratings = sub_df[f'{annot}_extraneous_gendered_word'].values
            extr_mean_data[exp][-1][it + 1] = np.mean(extr_ratings)
            extr_std_data[exp][-1][it + 1] = sem(extr_ratings)

            examples = sub_df[gen_col][sub_df[f'{annot}_extraneous_gendered_word']]
            n = min(len(examples), 1)
            examples = examples.sample(n=n, random_state=131719).values
            if n > 0:
                extr_samples[exp][-1][1].append(examples[0])

            annot_count_data[exp][-1][it + 1] = []
            for a in annots:
                annot_count_data[exp][-1][it + 1].append(np.count_nonzero(sub_df[f'{annot}_label_pronoun'].values == a))
            annot_count_data[exp][-1][it + 1] = np.array(annot_count_data[exp][-1][it + 1])

plt.rcParams.update({'font.size': 18})
# to change default colormap
plt.rcParams["image.cmap"] = "viridis"
# to change default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

print("Agreement pre-[MASK]")
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True, sharex=True, layout='constrained')
mean_df = pd.DataFrame(agr_mean_data['pre'], columns=['model'] + data_cols).sort_values('model')
std_df = pd.DataFrame(agr_std_data['pre'], columns=['model'] + data_cols).sort_values('model')

trans = [Affine2D().translate(-0.18, 0.0) + axs.transData, \
         Affine2D().translate(-0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.18, 0.0) + axs.transData]

for idx, pronoun in enumerate(data_cols):
    axs.errorbar(x = mean_df['model'], y = mean_df[pronoun], yerr=std_df[pronoun], fmt="o", label=pronoun, capsize=5, transform=trans[idx])

ident = [0, len(model_names) - 1]
axs.plot(ident, [1.0] * len(ident), linestyle='--', color='k')

axs.tick_params(axis='x', labelrotation=90)
axs.set_ylabel(r'Raw Agreement ($\uparrow$)')
axs.set_title('Human vs. Pre-[MASK] Gen')

plt.legend(bbox_to_anchor=(1.05, 1.05))
plt.tight_layout()
plt.savefig('plots/MISGENDERED-pre-human-agreement.pdf')

print("Agreement post-[MASK]")
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True, sharex=True, layout='constrained')
mean_df = pd.DataFrame(agr_mean_data['post'], columns=['model'] + data_cols).sort_values('model')
std_df = pd.DataFrame(agr_std_data['post'], columns=['model'] + data_cols).sort_values('model')

trans = [Affine2D().translate(-0.18, 0.0) + axs.transData, \
         Affine2D().translate(-0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.18, 0.0) + axs.transData]

for idx, pronoun in enumerate(data_cols):
    axs.errorbar(x = mean_df['model'], y = mean_df[pronoun], yerr=std_df[pronoun], fmt="o", label=pronoun, capsize=5, transform=trans[idx])

ident = [0, len(model_names) - 1]
axs.plot(ident, [1.0] * len(ident), linestyle='--', color='k')

axs.tick_params(axis='x', labelrotation=90)
axs.set_ylabel(r'Raw Agreement ($\uparrow$)')
axs.set_title('Human vs. Post-[MASK] Gen')

plt.legend(bbox_to_anchor=(1.05, 1.05))
plt.tight_layout()
plt.savefig('plots/MISGENDERED-post-human-agreement.pdf')

plt.rcParams.update({'font.size': 18})
# to change default colormap
plt.rcParams["image.cmap"] = "viridis"
# to change default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

print("Extraneous pre-[MASK]")
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True, sharex=True, layout='constrained')
mean_df = pd.DataFrame(extr_mean_data['pre'], columns=['model'] + data_cols).sort_values('model')
std_df = pd.DataFrame(extr_std_data['pre'], columns=['model'] + data_cols).sort_values('model')

trans = [Affine2D().translate(-0.18, 0.0) + axs.transData, \
         Affine2D().translate(-0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.18, 0.0) + axs.transData]

for idx, pronoun in enumerate(data_cols):
    axs.errorbar(x = mean_df['model'], y = mean_df[pronoun], yerr=std_df[pronoun], fmt="o", label=pronoun, capsize=5, transform=trans[idx])

ident = [0, len(model_names) - 1]
axs.plot(ident, [0.0] * len(ident), linestyle='--', color='k')

axs.tick_params(axis='x', labelrotation=90)
axs.set_ylabel(r'Extraneous Gendered Word Rate ($\downarrow$)')
axs.set_title('Pre-[MASK] Gen')

plt.legend()
plt.tight_layout()
plt.savefig('plots/MISGENDERED-pre-extraneous.pdf')

print("Extraneous post-[MASK]")
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True, sharex=True, layout='constrained')
mean_df = pd.DataFrame(extr_mean_data['post'], columns=['model'] + data_cols).sort_values('model')
std_df = pd.DataFrame(extr_std_data['post'], columns=['model'] + data_cols).sort_values('model')

trans = [Affine2D().translate(-0.18, 0.0) + axs.transData, \
         Affine2D().translate(-0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.18, 0.0) + axs.transData]

for idx, pronoun in enumerate(data_cols):
    axs.errorbar(x = mean_df['model'], y = mean_df[pronoun], yerr=std_df[pronoun], fmt="o", label=pronoun, capsize=5, transform=trans[idx])

ident = [0, len(model_names) - 1]
axs.plot(ident, [0.0] * len(ident), linestyle='--', color='k')

axs.tick_params(axis='x', labelrotation=90)
axs.set_ylabel(r'Extraneous Gendered Word Rate ($\downarrow$)')
axs.set_title('Post-[MASK] Gen')

plt.legend()
plt.tight_layout()
plt.savefig('plots/MISGENDERED-post-extraneous.pdf')

pre_df = pd.DataFrame(annot_count_data['pre'], columns=['model'] + data_cols).sort_values('model').set_index('model')
print(pre_df.values.sum(axis=0))
print()
print(pre_df.values.sum(axis=1))

post_df = pd.DataFrame(annot_count_data['post'], columns=['model'] + data_cols).sort_values('model').set_index('model')
print(post_df.values.sum(axis=0))
print()
print(post_df.values.sum(axis=1))


## Qualitative Examples (Extraneous Gendered Language)

random.seed(131719)
extr_samples_df = pd.DataFrame(extr_samples['pre'], columns=['', '\\textbf{Examples}']).sort_values('').set_index('')
sub_extr_samples_df = extr_samples_df[['\\textbf{Examples}']].map(lambda x: random.choice(x))

print(sub_extr_samples_df.to_latex(index=True, bold_rows=True, column_format='p{0.2\linewidth} | p{0.75\linewidth}'))

random.seed(131719)
extr_samples_df = pd.DataFrame(extr_samples['post'], columns=['', '\\textbf{Examples}']).sort_values('').set_index('')
sub_extr_samples_df = extr_samples_df[['\\textbf{Examples}']].map(lambda x: random.choice(x))

print(sub_extr_samples_df.to_latex(index=True, bold_rows=True, column_format='p{0.2\linewidth} | p{0.75\linewidth}'))


## Qualitative Examples (Disagreement)

random.seed(131719)
agr_samples_df = pd.DataFrame(agr_samples['pre'], columns=['', '\\textbf{Examples}']).sort_values('').set_index('')
sub_agr_samples_df = agr_samples_df[['\\textbf{Examples}']].map(lambda x: random.choice(x))

print(sub_agr_samples_df.to_latex(index=True, bold_rows=True, column_format='p{0.2\linewidth} | p{0.75\linewidth}'))

random.seed(131719)
agr_samples_df = pd.DataFrame(agr_samples['post'], columns=['', '\\textbf{Examples}']).sort_values('').set_index('')
sub_agr_samples_df = agr_samples_df[['\\textbf{Examples}']].map(lambda x: random.choice(x) if len(x) > 0 else '---')

print(sub_agr_samples_df.to_latex(index=True, bold_rows=True, column_format='p{0.2\linewidth} | p{0.75\linewidth}'))