#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONPATH=".:src:src/baselines"

python -u main.py \
  --env_name predator_prey \
  --model dgn \
  --env_graph 1 \
  --nagents 5 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.001 \
  --recurrent 1 \
  --seed 0 \
  | tee scripts/train_pp_medium_dgn.log
