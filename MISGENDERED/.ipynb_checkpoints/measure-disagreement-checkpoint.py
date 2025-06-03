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

import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from matplotlib.transforms import Affine2D

import pandas as pd
from statsmodels.stats.inter_rater import cohens_kappa
from scipy.stats import pearsonr, sem
from math import isclose

pre_cols = []
post_cols = []
for g in range(1, 6):
    pre_cols.append(f'greedy_pre_{str(g)}_correct')
for g in range(1, 6):
    post_cols.append(f'greedy_post_{str(g)}_correct')

def estimate_beta_dist_params(data):
    sample_mean = np.mean(data)
    sample_var = np.std(data, ddof=1) ** 2

    q = sample_mean * (1 - sample_mean)
    assert sample_var < q

    alpha = sample_mean * (q / sample_var - 1)
    beta = (1 - sample_mean) * (q / sample_var - 1)

    return alpha, beta, 0, 1

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

exclude_models = set([])

a_list = []
b_list = []
lbl_list = []

pre_kappa_data = []
post_kappa_data = []

pre_mcc_data = []
post_mcc_data = []

pre_agreement_mean_data = []
pre_agreement_std_data = []

post_agreement_mean_data = []
post_agreement_std_data = []

pre_variation_v1_data = []
pre_variation_v2_data = []
pre_variation_v3_data = []

post_variation_v1_data = []
post_variation_v2_data = []
post_variation_v3_data = []

data_cols = ["he", "she", "they", "xe"]

for model in sorted(model_names):
    if model in exclude_models:
        continue

    conv_model_name = model_name_map[model]
    
    pre_kappa_data.append([conv_model_name] + [None] * len(data_cols))
    pre_mcc_data.append([conv_model_name] + [None] * len(data_cols))
    pre_agreement_mean_data.append([conv_model_name] + [None] * len(data_cols))
    pre_agreement_std_data.append([conv_model_name] + [None] * len(data_cols))
    pre_variation_v1_data.append([conv_model_name] + [None] * len(data_cols))
    pre_variation_v2_data.append([conv_model_name] + [None] * len(data_cols))
    pre_variation_v3_data.append([conv_model_name] + [None] * len(data_cols))

    post_kappa_data.append([conv_model_name] + [None] * len(data_cols))
    post_mcc_data.append([conv_model_name] + [None] * len(data_cols))
    post_agreement_mean_data.append([conv_model_name] + [None] * len(data_cols))
    post_agreement_std_data.append([conv_model_name] + [None] * len(data_cols))
    post_variation_v1_data.append([conv_model_name] + [None] * len(data_cols))
    post_variation_v2_data.append([conv_model_name] + [None] * len(data_cols))
    post_variation_v3_data.append([conv_model_name] + [None] * len(data_cols))
    
    for it, f in enumerate(data_cols):
        f += '.csv'
        
        df = pd.read_csv("./out/" + model + "/" + f)
        
        print(model, f.split('.')[0])

        sample_std = np.round(np.std(df[pre_cols].values, axis=1), 3)
        stds, counts = np.unique(sample_std, return_counts=True)
        counts = counts.astype(float) / counts.sum()
        pre_variation_v1_data[-1][it + 1] = counts[0]
        pre_variation_v2_data[-1][it + 1] = counts[1]
        pre_variation_v3_data[-1][it + 1] = counts[2]
        pre_res = counts

        sample_std = np.round(np.std(df[post_cols].values, axis=1), 3)
        stds, counts = np.unique(sample_std, return_counts=True)
        counts = counts.astype(float) / counts.sum()
        post_variation_v1_data[-1][it + 1] = counts[0]
        post_variation_v2_data[-1][it + 1] = counts[1]
        post_variation_v3_data[-1][it + 1] = counts[2]
        post_res = counts

        print("Generation variation:", pre_res, post_res)

        prob_ratings = df['correct'].values
        col = pre_cols[0]
        pre_gen_ratings = df[col].values

        col = post_cols[0]
        post_gen_ratings = df[col].values

        contingency_table = pd.crosstab(prob_ratings, pre_gen_ratings)
        pre_kappa_results = cohens_kappa(contingency_table)
        pre_mcc_results = pearsonr(prob_ratings, pre_gen_ratings)

        contingency_table = pd.crosstab(prob_ratings, post_gen_ratings)
        post_kappa_results = cohens_kappa(contingency_table)
        post_mcc_results = pearsonr(prob_ratings, post_gen_ratings)

        pre_res = f'${pre_kappa_results.kappa:.3f}' + \
                f' \\pm {(pre_kappa_results.kappa_upp - pre_kappa_results.kappa_low) / 2:.3f}$'
        assert isclose(pre_kappa_results.kappa, (pre_kappa_results.kappa_upp + pre_kappa_results.kappa_low) / 2)
        pre_kappa_data[-1][it + 1] = pre_res

        post_res = f'${post_kappa_results.kappa:.3f}' + \
                f' \\pm {(post_kappa_results.kappa_upp - post_kappa_results.kappa_low) / 2:.3f}$'
        assert isclose(post_kappa_results.kappa, (post_kappa_results.kappa_upp + post_kappa_results.kappa_low) / 2)
        post_kappa_data[-1][it + 1] = post_res

        print("Cohen's kappa:", pre_res, post_res)

        low = pre_mcc_results.confidence_interval(confidence_level=0.95).low
        high = pre_mcc_results.confidence_interval(confidence_level=0.95).high
        pre_res = f'${pre_mcc_results.statistic:.3f}$ $[{low:.3f}, {high:.3f}]$'
        pre_mcc_data[-1][it + 1] = pre_res

        low = post_mcc_results.confidence_interval(confidence_level=0.95).low
        high = post_mcc_results.confidence_interval(confidence_level=0.95).high
        post_res = f'${post_mcc_results.statistic:.3f}$ $[{low:.3f}, {high:.3f}]$'
        post_mcc_data[-1][it + 1] = post_res

        print("MCC:", pre_res, post_res)

        pre_agreement_scores = (prob_ratings == pre_gen_ratings)
        post_agreement_scores = (prob_ratings == post_gen_ratings)

        pre_agreement_mean_data[-1][it + 1] = np.mean(pre_agreement_scores)
        pre_agreement_std_data[-1][it + 1] = sem(pre_agreement_scores)
        pre_res = f'{pre_agreement_mean_data[-1][it + 1]:.3f}' + ' ± ' + f'{pre_agreement_std_data[-1][it + 1]:.3f}'

        post_agreement_mean_data[-1][it + 1] = np.mean(post_agreement_scores)
        post_agreement_std_data[-1][it + 1] = sem(post_agreement_scores)
        post_res = f'{post_agreement_mean_data[-1][it + 1]:.3f}' + ' ± ' + f'{post_agreement_std_data[-1][it + 1]:.3f}'
        
        print("Agreement score:", pre_res, post_res)

        disagr = df['correct'].values * (1 - df['greedy_pre_correct'].values) + (1 - df['correct'].values) * df['greedy_pre_correct'].values
        a, b, loc, scale = estimate_beta_dist_params(disagr)
        print("Pre-[MASK] generative beta params:", a, b, loc, scale)
        a_list.append(a)
        b_list.append(b)
        lbl_list.append(conv_model_name + " " + f.split('.')[0] + " Pre-[MASK]")

        disagr = df['correct'].values * (1 - df['greedy_post_correct'].values) + (1 - df['correct'].values) * df['greedy_post_correct'].values
        a, b, loc, scale = estimate_beta_dist_params(disagr)
        print("Post-[MASK] generative beta params:", a, b, loc, scale)
        a_list.append(a)
        b_list.append(b)
        lbl_list.append(conv_model_name + " " + f.split('.')[0] + " Post-[MASK]")
        
        print()

plt.rcParams.update({'font.size': 18})

exps = ['Pre-[MASK]', 'Post-[MASK]']

cmap = cm.get_cmap('tab20c')

scaled = np.arange(len(model_names)) / len(model_names)
colors = [cmap(x) for x in scaled]
model_txts = [model_name_map[model] for model in model_names]

pro_txts = ["he", "she", "they", "xe"]
pro_markers = [f"${x[0]}$" for x in pro_txts]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 7), sharey=True, sharex=True, layout='constrained')

for it, exp in enumerate(exps):

    selected_markers = set()
    selected_colors = set()

    for i, txt in enumerate(lbl_list):
        if exp in txt:
            for p, m in zip(pro_txts, pro_markers):
                if " " + p in txt:
                    marker = m
                    selected_markers.add((p, m))
                    break

            for s, c in zip(model_txts, colors):
                if s.lower() in txt.lower():
                    color = c
                    selected_colors.add((s.replace('-4bit', ''), c))
                    break
            
            axs[it].scatter(b_list[i], a_list[i], color=color, marker=marker, s=100)
                
    axs[it].set_xlabel(r'$\beta$')
    axs[it].set_ylabel(r'$\alpha$')
    axs[it].set_title('Prob vs. ' + exp + ' Gen')
    axs[it].set_box_aspect(1)
    axs[it].set_xlim(-0.1, 3.0)
    axs[it].set_ylim(-0.1, 3.0)

    ident = [-0.1, 3.0]
    axs[it].plot(ident, ident, linestyle='--', color='k')
    axs[it].plot(ident, [1.0, 1.0], linestyle='--', color='k')
    axs[it].plot([1.0, 1.0], ident, linestyle='--', color='k')

    f = lambda m,c: axs[it].plot([],[],marker=m, color=c, ls="none")[0]

    selected_colors = sorted(list(selected_colors), key=lambda x: x[0])
    selected_markers = sorted(list(selected_markers), key=lambda x: x[0])
    
    handles = [f("s", c) for (s, c) in selected_colors]
    handles += [f(m, "k") for (p, m) in selected_markers]

    labels = [s for (s, c) in selected_colors] + [p for (p, m) in selected_markers]
    axs[-1].legend(handles, labels, framealpha=0.75, bbox_to_anchor=(1.1, 1.05))

plt.tight_layout()
plt.savefig('plots/MISGENDERED-beta.pdf')

formatted_cols = [r'\texttt{he}', r'\texttt{she}', r'\texttt{they}', r'\texttt{xe}']

print("Cohen's kappa pre-[MASK]")
df = pd.DataFrame(pre_kappa_data, columns=[''] + formatted_cols)
df = df.set_index('')
print(df.to_latex(index=True, bold_rows=True, column_format='ccccc'))

print("Cohen's kappa post-[MASK]")
df = pd.DataFrame(post_kappa_data, columns=[''] + formatted_cols)
df = df.set_index('')
print(df.to_latex(index=True, bold_rows=True, column_format='ccccc'))

formatted_cols = [r'\texttt{he}', r'\texttt{she}', r'\texttt{they}', r'\texttt{xe}']

print("MCC pre-[MASK]")
df = pd.DataFrame(pre_mcc_data, columns=[''] + formatted_cols)
df = df.set_index('')
print(df.to_latex(index=True, bold_rows=True, column_format='ccccc'))

print("MCC post-[MASK]")
df = pd.DataFrame(post_mcc_data, columns=[''] + formatted_cols)
df = df.set_index('')
print(df.to_latex(index=True, bold_rows=True, column_format='ccccc'))

plt.rcParams.update({'font.size': 18})
# to change default colormap
plt.rcParams["image.cmap"] = "viridis"
# to change default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

print("Agreement pre-[MASK]")
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True, sharex=True, layout='constrained')

mean_df = pd.DataFrame(pre_agreement_mean_data, columns=['model'] + data_cols)
std_df = pd.DataFrame(pre_agreement_std_data, columns=['model'] + data_cols)

trans = [Affine2D().translate(-0.18, 0.0) + axs.transData, \
         Affine2D().translate(-0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.18, 0.0) + axs.transData]

for idx, pronoun in enumerate(data_cols):
    axs.errorbar(x = mean_df['model'], y = mean_df[pronoun], markersize=7, yerr=std_df[pronoun], fmt="o", label=pronoun, capsize=5, transform=trans[idx])

ident = [0, len(model_names) - 1]
axs.plot(ident, [1.0] * len(ident), linestyle='--', color='k')

axs.tick_params(axis='x', labelrotation=45)
axs.set_ylabel(r'Raw Agreement ($\uparrow$)')
axs.set_ylim(0,None)
axs.set_title('Prob vs. Pre-[MASK] Gen')

plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('plots/MISGENDERED-pre-agreement.pdf')

print("Agreement post-[MASK]")
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True, sharex=True, layout='constrained')

mean_df = pd.DataFrame(post_agreement_mean_data, columns=['model'] + data_cols)
std_df = pd.DataFrame(post_agreement_std_data, columns=['model'] + data_cols)

trans = [Affine2D().translate(-0.18, 0.0) + axs.transData, \
         Affine2D().translate(-0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.18, 0.0) + axs.transData]

for idx, pronoun in enumerate(data_cols):
    axs.errorbar(x = mean_df['model'], y = mean_df[pronoun], markersize=7, yerr=std_df[pronoun],
                 fmt="o", label=pronoun, capsize=5, transform=trans[idx])

ident = [0, len(model_names) - 1]
axs.plot(ident, [1.0] * len(ident), linestyle='--', color='k')

axs.tick_params(axis='x', labelrotation=45)
axs.set_ylabel(r'Raw Agreement ($\uparrow$)')
axs.set_ylim(0,None)
axs.set_title('Prob vs. Post-[MASK] Gen')

plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('plots/MISGENDERED-post-agreement.pdf')

plt.rcParams.update({'font.size': 18})
# to change default colormap
plt.rcParams["image.cmap"] = "viridis"
# to change default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

print("Generation variation pre-[MASK]")
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True, sharex=True, layout='constrained')

v1_df = pd.DataFrame(pre_variation_v1_data, columns=['model'] + data_cols)
v2_df = pd.DataFrame(pre_variation_v2_data, columns=['model'] + data_cols)
v3_df = pd.DataFrame(pre_variation_v3_data, columns=['model'] + data_cols)

trans = [Affine2D().translate(-0.24, 0.0) + axs.transData, \
         Affine2D().translate(-0.08, 0.0) + axs.transData, \
         Affine2D().translate(0.08, 0.0) + axs.transData, \
         Affine2D().translate(0.24, 0.0) + axs.transData]
shifts = [-0.24, -0.08, 0.08, 0.24]

for idx, pronoun in enumerate(data_cols):
    l1, l2, l3 = '', '', ''
    if idx == 0:
        l1, l2, l3 = r'$\sigma = 0.0$', r'$\sigma = 0.4$', r'$\sigma = 0.49$'
    
    axs.bar(x = v1_df['model'], height = v1_df[pronoun], color="#440154FF", width=0.1, label=l1, transform=trans[idx])
    axs.bar(x = v2_df['model'], height = v2_df[pronoun], bottom=v1_df[pronoun], color="#2A788EFF", width=0.1, label=l2, transform=trans[idx])
    bars = axs.bar(x = v3_df['model'], height = v3_df[pronoun], bottom=v1_df[pronoun] + v2_df[pronoun], color="#7AD151FF", width=0.1, label=l3, transform=trans[idx])
    for bar in bars:
        axs.text(bar.get_x() + shifts[idx] + 0.05, 1, pronoun[0], ha='center', va='bottom', fontsize=12)

axs.tick_params(axis='x', labelrotation=45)
axs.set_ylabel(r'Normalized Frequency')
axs.set_title('Pre-[MASK] Gen')

plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('plots/MISGENDERED-pre-gen-variation.pdf')

print("Generation variation post-[MASK]")
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True, sharex=True, layout='constrained')

v1_df = pd.DataFrame(post_variation_v1_data, columns=['model'] + data_cols)
v2_df = pd.DataFrame(post_variation_v2_data, columns=['model'] + data_cols)
v3_df = pd.DataFrame(post_variation_v3_data, columns=['model'] + data_cols)

trans = [Affine2D().translate(-0.24, 0.0) + axs.transData, \
         Affine2D().translate(-0.08, 0.0) + axs.transData, \
         Affine2D().translate(0.08, 0.0) + axs.transData, \
         Affine2D().translate(0.24, 0.0) + axs.transData]
shifts = [-0.24, -0.08, 0.08, 0.24]

for idx, pronoun in enumerate(data_cols):
    l1, l2, l3 = '', '', ''
    if idx == 0:
        l1, l2, l3 = r'$\sigma = 0.0$', r'$\sigma = 0.4$', r'$\sigma = 0.49$'
        
    axs.bar(x = v1_df['model'], height = v1_df[pronoun], color="#440154FF", width=0.1, label=l1, transform=trans[idx])
    axs.bar(x = v2_df['model'], height = v2_df[pronoun], bottom=v1_df[pronoun], color="#2A788EFF", width=0.1, label=l2, transform=trans[idx])
    bars = axs.bar(x = v3_df['model'], height = v3_df[pronoun], bottom=v1_df[pronoun] + v2_df[pronoun], color="#7AD151FF", width=0.1, label=l3, transform=trans[idx])
    for bar in bars:
        axs.text(bar.get_x() + shifts[idx] + 0.05, 1, pronoun[0], ha='center', va='bottom', fontsize=12)

axs.tick_params(axis='x', labelrotation=45)
axs.set_ylabel(r'Normalized Frequency')
axs.set_title('Post-[MASK] Gen')

plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('plots/MISGENDERED-post-gen-variation.pdf')