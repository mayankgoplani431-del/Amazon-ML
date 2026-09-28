#!/bin/bash
# Full pipeline: data -> normalization -> blocking -> pruning -> features -> models -> decision -> output/.
# Put the competition dataset in data/train and data/test first (see README). ~7-8 h on a 16 GB laptop.
set -e
cd "$(dirname "$0")/../src"
export PYTHONIOENCODING=utf-8
python prep.py train test                 # 1. normalize all 6 files (+ learn transliteration dictionary from train)
python candidates.py train                # 2. multi-pass blocking on train
STAGES=2 python run_train.py EXP_NAME 20  # 3. stage-0 pruning, features, stage-1/2 LightGBM, OOF validation scorecard
python entity_tune.py                     # 4. entity-level empty/count decision model (optional, small gain)
python candidates.py test                 # 5. blocking on test
python run_test.py ../output 20           # 6. score test, write output/matching_results.tsv + candidate_pairs.tsv
python check_output.py ../output ../data/test   # 7. independent validation (also run the official validator)
