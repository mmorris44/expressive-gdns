#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONPATH=".:src:src/baselines"

python -u main.py \
  --env_name traffic_junction \
  --model dgn \
  --env_graph 1 \
  --nagents 5 \
  --dim 6 \
  --max_steps 20 \
  --difficulty easy \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 2000 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --recurrent 1 \
  --seed 0 \
  --rni 1 \
  | tee scripts/train_tj_easy_dgn.log
