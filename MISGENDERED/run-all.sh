for model in meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-70B mistralai/Mixtral-8x7B-v0.1 mistralai/Mixtral-8x22B-v0.1 allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-13B; do
    for pronoun in he she they xe; do
        python eval.py --pronoun $pronoun --model $model
    done 
done