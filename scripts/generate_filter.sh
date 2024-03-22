#!/bin/bash

python rank_score.py \
  --taxo_name semeval_sci \
  --model_path allenai/scibert_scivocab_uncased \
  --save_path ./filter/
