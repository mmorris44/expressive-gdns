#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONPATH=".:src:src/baselines"

python -u main.py \
  --env_name drone_scatter \
  --model magic \
  --nagents 4 \
  --dim 20 \
  --max_steps 40 \
  --difficulty easy \
  --nprocesses 1 \
  --num_epochs 1500 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --directed 1 \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --self_loop_type 1 \
  --message_decoder 1 \
  --seed 0 \
  --rni 0 \
  --display 0 \
  --greedy_a2c_eval 1 \
  | tee scripts/train_ds_easy_magic.log
