# Chain-of-Layer

Code for Chain-of-Layer

## Ensemble-based Ranking Filter

```
bash ./scripts/generate_filter.sh
```

- `--taxo_name`: ['wordnet', 'semeval_sci', 'wiki', 'dblp']
- `--model_path`: allenai/scibert_scivocab_uncased
- `--save_path`: the save path for the generated filter score JSON file.

## Hierarchical Format Taxonomy Induction Instruction (HF)

```
bash ./scripts/run_HF.sh
```

- `--openai_key`: your openai key
- `--taxo_name`: ['wordnet', 'semeval_sci', 'wiki', 'dblp']
- `--model`: ['gpt-3.5-turbo-16k', 'gpt-4-turbo-preview']
- `--numofExamples`: 0 for zero-shot and 5 for five shot
- `--run`: set to `False` to only evaluate the saved results
- `--save_path_model_response`: the save path of model response
- `--demo_path`: the path of the demo
- `--ChainofLayers & iteratively`: set to `False` to only use Hierarchical Format Taxonomy Induction Instruction

## Chain-of-Layer (CoL)

```
bash ./scripts/run_CoL.sh
```

- `--openai_key`: your openai key
- `--taxo_name`: ['wordnet', 'semeval_sci', 'wiki', 'dblp']
- `--model`: ['gpt-3.5-turbo-16k', 'gpt-4-turbo-preview']
- `--numofExamples`: 0 for zero-shot and 5 for five shot
- `--run`: set to `False` to only evaluate the saved results
- `--save_path_model_response`: the save path of model response
- `--demo_path`: the path of the demo
- `--ChainofLayers & iteratively`: set to `True` to use Chain-of-Layer
- `--filter_mode`: default `lm_score_ensemble`
- `--filter_model`: default `scibert_scivocab_uncased`
- `--filter_topk`: Top-K filter

## CoL-Zero

```
bash ./scripts/gen_demo_CoL_zero.sh
```

- `--taxo_name`: ['wordnet', 'semeval_sci', 'wiki', 'dblp']
- `--save_path_model_response`: the save path of raw demo
- `--save_path`: the save path of the processed demo
- `--mode`: default 1

```
bash ./scripts/run_CoL_zero.sh
```
