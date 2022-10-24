#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONPATH=".:src:src/baselines"

python -u main.py \
  --env_name drone_scatter \
  --model dgn \
  --env_graph 1 \
  --nagents 4 \
  --dim 20 \
  --max_steps 20 \
  --difficulty easy \
  --num_epochs 2000 \
  --seed 0 \
  --rni 0 \
  --display 0 \
  --comm_passes 4 \
  --share_weights 1 \
  | tee scripts/train_ds_easy_dgn.log
