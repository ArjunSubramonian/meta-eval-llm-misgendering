# Agree to Disagree? A Meta-Evaluation of LLM Misgendering

This repository contains the code and data for the paper.

## Code

- `*/run-all.sh`, `*/eval.py`: main scripts that run parallel probability- and generation-based evaluations for all models and ground-truth pronouns for each dataset
- `*/sample-for-manual-verification.py`: sample model generations for human annotation
- `*/measure-disagreement.py`: compute disagreement metrics and generate model evaluation plots presented in the paper
- `*/manual-annotations.py`: generate human evaluation plots and sample qualitative examples presented in the paper
- `*/measure-repetition.py`: compute repetitiveness score of model generations
- `TANGO/neutralize.py`, `TANGO/deneutralize.py`, `TANGO/constants.py`: rewrite model generations with different pronouns
- `collate_samples.py`: collate model generations sampled for human annotation across pronouns and pre-/post-MASK settings

## Data

- `*/out.zip`: password-protected zip file that contains results of probability- and generation-based evaluations for all models and ground-truth pronouns for each dataset
- `*/samples.zip`: password-protected zip file that contains sampled model generations for human annotation

The password for all zip files is: `lepoissonsteve232729`. Please do not publicly upload the raw contents of these zip files anywhere, as they contain model-generated instances of misgendering.

All human annotations can be found [here](https://docs.google.com/spreadsheets/d/14a9qqtU86_AFOwqaqy63O-SJKr_fXAOesezt-DDvs10/edit).