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

prob_cols = []
gen_cols = []
for g in range(1, 6):
    prob_cols.append(f'olg_template_{str(g)}_correct')
for g in range(1, 6):
    gen_cols.append(f'olg_{str(g)}_correct')
    
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

kappa_data = []
mcc_data = []

agreement_mean_data = []
agreement_count_data = []
agreement_std_data = []

prob_variation_v1_data = []
prob_variation_v2_data = []
prob_variation_v3_data = []

gen_variation_v1_data = []
gen_variation_v2_data = []
gen_variation_v3_data = []

failure_mean_data = []
failure_std_data = []

data_cols = ["he", "she", "they", "xe"]

for model in sorted(model_names):
    if model in exclude_models:
        continue

    conv_model_name = model_name_map[model]
    
    kappa_data.append([conv_model_name] + [None] * len(data_cols))
    mcc_data.append([conv_model_name] + [None] * len(data_cols))
    
    agreement_mean_data.append([conv_model_name] + [None] * len(data_cols))
    agreement_count_data.append([conv_model_name] + [None] * len(data_cols))
    agreement_std_data.append([conv_model_name] + [None] * len(data_cols))
    
    prob_variation_v1_data.append([conv_model_name] + [None] * len(data_cols))
    prob_variation_v2_data.append([conv_model_name] + [None] * len(data_cols))
    prob_variation_v3_data.append([conv_model_name] + [None] * len(data_cols))
    
    gen_variation_v1_data.append([conv_model_name] + [None] * len(data_cols))
    gen_variation_v2_data.append([conv_model_name] + [None] * len(data_cols))
    gen_variation_v3_data.append([conv_model_name] + [None] * len(data_cols))

    failure_mean_data.append([conv_model_name] + [None] * len(data_cols))
    failure_std_data.append([conv_model_name] + [None] * len(data_cols))
    
    for it, f in enumerate(data_cols):
        f += '.csv'
        
        df = pd.read_csv("./out/" + model + "/" + f)
        print(model, f.split('.')[0])

        prob_data = df[prob_cols].values
        mask = np.isnan(prob_data)

        failure_mean_data[-1][it + 1] = np.mean(mask.mean(axis=1))       
        failure_std_data[-1][it + 1] = sem(mask.mean(axis=1))

        prob_data = prob_data[~mask.all(axis=1)]
        prob_data = np.ma.array(prob_data, mask=np.isnan(prob_data))
        gen_data = df[gen_cols].values

        sample_std = np.std(prob_data, axis=1).data
        prob_variation_v1_data[-1][it + 1] = np.mean(sample_std)
        prob_variation_v2_data[-1][it + 1] = np.std(sample_std)
        prob_variation_v3_data[-1][it + 1] = np.std(sample_std)
        prob_res = (prob_variation_v1_data[-1][it + 1], prob_variation_v2_data[-1][it + 1], prob_variation_v3_data[-1][it + 1])

        sample_std = np.round(np.std(gen_data, axis=1), 3)
        stds, counts = np.unique(sample_std, return_counts=True)
        counts = counts.astype(float) / counts.sum()
        gen_variation_v1_data[-1][it + 1] = counts[0]
        gen_variation_v2_data[-1][it + 1] = counts[1]
        gen_variation_v3_data[-1][it + 1] = counts[2]
        gen_res = counts

        print("Generation variation:", prob_res, gen_res)

        pc = prob_cols[0]
        gc = gen_cols[0]
        prob_ratings = df[pc].values
        gen_ratings = df[gc].values

        mask = np.isnan(prob_ratings)
        prob_ratings = prob_ratings[~mask]
        gen_ratings = gen_ratings[~mask]

        contingency_table = pd.crosstab(prob_ratings, gen_ratings)
        kappa_results = cohens_kappa(contingency_table)
        mcc_results = pearsonr(prob_ratings, gen_ratings)

        res = f'${kappa_results.kappa:.3f}' + \
                f' \\pm {(kappa_results.kappa_upp - kappa_results.kappa_low) / 2:.3f}$'
        assert isclose(kappa_results.kappa, (kappa_results.kappa_upp + kappa_results.kappa_low) / 2)
        kappa_data[-1][it + 1] = res

        print("Cohen's kappa:", res)

        low = mcc_results.confidence_interval(confidence_level=0.95).low
        high = mcc_results.confidence_interval(confidence_level=0.95).high
        res = f'${mcc_results.statistic:.3f}$ $[{low:.3f}, {high:.3f}]$'
        mcc_data[-1][it + 1] = res

        print("MCC:", res)

        agreement_scores = (prob_ratings == gen_ratings)
        agreement_mean_data[-1][it + 1] = np.mean(agreement_scores)
        agreement_count_data[-1][it + 1] = prob_ratings.shape[0]
        agreement_std_data[-1][it + 1] = sem(agreement_scores)
        res = f'{agreement_mean_data[-1][it + 1]:.3f}' + ' ± ' + f'{agreement_std_data[-1][it + 1]:.3f}'
        
        print("Agreement score:", res)

        prob_correct = df['prob_correct'].values
        mask = np.isnan(prob_correct)
        prob_correct = prob_correct[~mask]
        gen_correct = df['gen_correct'].values[~mask]

        disagr = prob_correct * (1 - gen_correct) + (1 - prob_correct) * gen_correct
        a, b, loc, scale = estimate_beta_dist_params(disagr)
        print("Beta params:", a, b, loc, scale)
        a_list.append(a)
        b_list.append(b)
        lbl_list.append(conv_model_name + " " + f.split('.')[0])
        
        print()

plt.rcParams.update({'font.size': 18})
# to change default colormap
plt.rcParams["image.cmap"] = "viridis"
# to change default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True, sharex=True, layout='constrained')

mean_df = pd.DataFrame(failure_mean_data, columns=['model'] + data_cols)
std_df = pd.DataFrame(failure_std_data, columns=['model'] + data_cols)

trans = [Affine2D().translate(-0.18, 0.0) + axs.transData, \
         Affine2D().translate(-0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.18, 0.0) + axs.transData]

for idx, pronoun in enumerate(data_cols):
    axs.errorbar(x = mean_df['model'], y = mean_df[pronoun], markersize=7, yerr=std_df[pronoun], fmt="o", label=pronoun, capsize=5, transform=trans[idx])

ident = [0, len(model_names) - 1]
axs.plot(ident, [0.0] * len(ident), linestyle='--', color='k')

axs.tick_params(axis='x', labelrotation=45)
axs.set_ylabel(r'Failure Rate ($\downarrow$)')
axs.set_ylim(None,1.05)

plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('plots/TANGO-failure.pdf')

plt.rcParams.update({'font.size': 18})

cmap = cm.get_cmap('tab20c')

scaled = np.arange(len(model_names)) / len(model_names)
colors = [cmap(x) for x in scaled]
model_txts = [model_name_map[model] for model in model_names]

pro_txts = ["he", "she", "they", "xe"]
pro_markers = [f"${x[0]}$" for x in pro_txts]

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(18, 7), sharey=True, sharex=True, layout='constrained')

selected_markers = set()
selected_colors = set()

for i, txt in enumerate(lbl_list):
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
        
    axs.scatter(b_list[i], a_list[i], color=color, marker=marker, s=100)
            
axs.set_xlabel(r'$\beta$')
axs.set_ylabel(r'$\alpha$')
axs.set_title('Prob vs. Gen')
axs.set_box_aspect(1)
axs.set_xlim(-0.1, 4.0)
axs.set_ylim(-0.1, 4.0)

ident = [-0.1, 4.0]
axs.plot(ident, ident, linestyle='--', color='k')
axs.plot(ident, [1.0, 1.0], linestyle='--', color='k')
axs.plot([1.0, 1.0], ident, linestyle='--', color='k')

f = lambda m,c: axs.plot([],[],marker=m, color=c, ls="none")[0]

selected_colors = sorted(list(selected_colors), key=lambda x: x[0])
selected_markers = sorted(list(selected_markers), key=lambda x: x[0])

handles = [f("s", c) for (s, c) in selected_colors]
handles += [f(m, "k") for (p, m) in selected_markers]

labels = [s for (s, c) in selected_colors] + [p for (p, m) in selected_markers]
axs.legend(handles, labels, framealpha=0.75, bbox_to_anchor=(1.1, 1.05))

plt.tight_layout()
plt.savefig('plots/TANGO-beta.pdf')

formatted_cols = [r'\texttt{he}', r'\texttt{she}', r'\texttt{they}', r'\texttt{xe}']

print("Cohen's kappa")
df = pd.DataFrame(kappa_data, columns=[''] + formatted_cols)
df = df.set_index('')
print(df.to_latex(index=True, bold_rows=True, column_format='ccccc'))

formatted_cols = [r'\texttt{he}', r'\texttt{she}', r'\texttt{they}', r'\texttt{xe}']

print("MCC")
df = pd.DataFrame(mcc_data, columns=[''] + formatted_cols)
df = df.set_index('')
print(df.to_latex(index=True, bold_rows=True, column_format='ccccc'))

plt.rcParams.update({'font.size': 18})
# to change default colormap
plt.rcParams["image.cmap"] = "viridis"
# to change default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True, sharex=True, layout='constrained')

mean_df = pd.DataFrame(agreement_mean_data, columns=['model'] + data_cols)
std_df = pd.DataFrame(agreement_std_data, columns=['model'] + data_cols)

trans = [Affine2D().translate(-0.18, 0.0) + axs.transData, \
         Affine2D().translate(-0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.18, 0.0) + axs.transData]

for idx, pronoun in enumerate(data_cols):
    axs.errorbar(x = mean_df['model'], y = mean_df[pronoun], markersize=7, yerr=std_df[pronoun], fmt="o", label=pronoun, capsize=5, transform=trans[idx])

ident = [0, len(model_names) - 1]
axs.plot(ident, [1.0] * len(ident), linestyle='--', color='k')

axs.tick_params(axis='x', labelrotation=45)
axs.set_ylim(0,1.05)
axs.set_ylabel(r'Raw Agreement ($\uparrow$)')
axs.set_title('Prob vs. Gen')

plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('plots/TANGO-agreement.pdf')

print("Generative variation")
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True, sharex=True, layout='constrained')

v1_df = pd.DataFrame(gen_variation_v1_data, columns=['model'] + data_cols)
v2_df = pd.DataFrame(gen_variation_v2_data, columns=['model'] + data_cols)
v3_df = pd.DataFrame(gen_variation_v3_data, columns=['model'] + data_cols)

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
axs.set_title('Gen')

plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('plots/TANGO-gen-variation.pdf')

print("Probablistic variation")
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True, sharex=True, layout='constrained')

median_df = pd.DataFrame(prob_variation_v1_data, columns=['model'] + data_cols)
q25_df = pd.DataFrame(prob_variation_v2_data, columns=['model'] + data_cols)
q75_df = pd.DataFrame(prob_variation_v3_data, columns=['model'] + data_cols)

trans = [Affine2D().translate(-0.18, 0.0) + axs.transData, \
         Affine2D().translate(-0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.06, 0.0) + axs.transData, \
         Affine2D().translate(0.18, 0.0) + axs.transData]

for idx, pronoun in enumerate(data_cols):
    axs.errorbar(x = median_df['model'], y = median_df[pronoun], markersize=7,
                 yerr=q25_df[pronoun], fmt='o', label=pronoun, capsize=5, transform=trans[idx])

ident = [0, len(model_names) - 1]
axs.plot(ident, [0.0] * len(ident), linestyle='--', color='k')

axs.tick_params(axis='x', labelrotation=45)
axs.set_ylabel(r'Instance-Level Variation ($\downarrow$)')
axs.set_title('Prob')
axs.set_ylim((0, 1.05))

plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('plots/TANGO-prob-variation.pdf')