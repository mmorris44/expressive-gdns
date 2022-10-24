#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONPATH=".:src:src/baselines"

python -u main.py \
  --env_name traffic_junction \
  --model dgn \
  --env_graph 1 \
  --nagents 10 \
  --dim 14 \
  --max_steps 40 \
  --difficulty medium \
  --vision 1 \
  --num_epochs 2000 \
  --recurrent 1 \
  --seed 0 \
  --rni 1 \
  --display 0 \
  | tee scripts/train_tj_medium_dgn.log
