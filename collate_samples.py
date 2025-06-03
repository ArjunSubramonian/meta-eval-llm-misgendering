from glob import glob
from pathlib import Path
import pandas as pd

for model_p in glob('./MISGENDERED/samples/*'):
    model = Path(model_p).name
    pre_dfs = []
    post_dfs = []
    for f in glob(f'{model_p}/*'):
        pronoun, pre_post, _ = Path(f).name.split('_', 2)
        df = pd.read_csv(f, index_col=0)
        df['pronoun'] = pronoun
        df['model'] = model
        if pre_post == 'pre':
            pre_dfs.append(df)
        elif pre_post == 'post':
            post_dfs.append(df)
    print(f'{model_p}/MISGENDERED_{model}_pre.csv')
    pd.concat(pre_dfs).to_csv(f'{model_p}/MISGENDERED_{model}_pre.csv', index=None)
    pd.concat(post_dfs).to_csv(f'{model_p}/MISGENDERED_{model}_post.csv', index=None)

for model_p in glob('./RUFF/samples/*'):
    model = Path(model_p).name
    pre_dfs = []
    post_dfs = []
    for f in glob(f'{model_p}/*'):
        pronoun, pre_post, _ = Path(f).name.split('_', 2)
        df = pd.read_csv(f, index_col=0)
        df['pronoun'] = pronoun
        df['model'] = model
        if pre_post == 'pre':
            pre_dfs.append(df)
        elif pre_post == 'post':
            post_dfs.append(df)
    print(f'{model_p}/RUFF_{model}_pre.csv')
    pd.concat(pre_dfs).to_csv(f'{model_p}/RUFF_{model}_pre.csv', index=None)
    pd.concat(post_dfs).to_csv(f'{model_p}/RUFF_{model}_post.csv', index=None)

