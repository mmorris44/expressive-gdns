#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONPATH=".:src:src/baselines"

python -u main.py \
  --env_name box_pushing \
  --model magic \
  --directed 1 \
  --message_decoder 1 \
  --env_graph 1 \
  --nagents 10 \
  --dim 12 \
  --max_steps 20 \
  --difficulty easy \
  --num_epochs 2000 \
  --recurrent 1 \
  --seed 0 \
  --rni 0.75 \
  --display 0 \
  --vision 1 \
  --comm_passes 4 \
  --imitation 1 \
  --num_imitation_experiences 100 \
  --num_normal_experiences 500 \
  | tee scripts/train_bp_easy_magic.log
