#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONPATH=".:src:src/baselines"

python -u main.py \
  --env_name traffic_junction \
  --env_graph 0 \
  --model magic \
  --nagents 5 \
  --dim 6 \
  --max_steps 20 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --difficulty easy \
  --vision 1 \
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
  --curr_start 0 \
  --curr_end 0 \
  --seed 1 \
  --rni 1 \
  --comm_passes 4 \
  --greedy_a2c_eval 0 \
  | tee scripts/train_tj_easy_magic.log
