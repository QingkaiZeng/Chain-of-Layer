#!/bin/bash

python infer.py \
  --openai_key your_openai_key \
  --taxo_name semeval_sci \
  --model gpt-4-1106-preview \
  --numofExamples 5 \
  --run True \
  --save_path_model_response ./results/taxo_ChainofLayers_filter_init/ \
  --demo_path ./demos/demo_wordnet_train/ \
  --ChainofLayers True \
  --iteratively True \
  --filter_mode lm_score_ensemble \
  --filter_model scibert_scivocab_uncased \
  --filter_topk 10