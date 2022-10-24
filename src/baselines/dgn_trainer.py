import copy
import random
from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn

from envs.imitation_learning import *
from utils import *
from action_utils import *
from RNI_utils import augment_state
from DGN import DGN
from buffer import ReplayBuffer

Transition = namedtuple('Transition',
                        ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                         'reward', 'misc', 'env_graph', 'next_env_graph'))


class DGNTrainer(object):
    def __init__(self,
                 args,
                 policy_net: DGN,
                 env
                 ):
        self.args = args
        self.model = policy_net
        self.model_tar = DGN(args, policy_net.num_inputs)
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.Adam(policy_net.parameters(), lr=args.lrate)
        self.params = [p for p in self.model.parameters()]
        self.complete_env_graph = get_complete_env_graph(args.nagents)

        epsilon_start = args.epsilon_start
        self.epsilon_min = args.epsilon_min
        self.epsilon_step = args.epsilon_step
        self.epsilon = epsilon_start

        buffer_capacity = args.buffer_capacity
        self.buff = ReplayBuffer(buffer_capacity)

        # For DGN training
        self.O = np.ones((args.dgn_batch_size, args.nagents, policy_net.num_inputs))
        self.Next_O = np.ones((args.dgn_batch_size, args.nagents, policy_net.num_inputs))
        self.Matrix = np.ones((args.dgn_batch_size, args.nagents, args.nagents))
        self.Next_Matrix = np.ones((args.dgn_batch_size, args.nagents, args.nagents))

        # For imitation learning
        if self.args.imitation:
            self.experience = get_experience(env, args.env_name)

    def get_episode(self,
                    epoch
                    ):
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_step

        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state, env_graph = self.env.reset(epoch)
        else:
            state, env_graph = self.env.reset()

        # Possibly add RNI to state
        compute_state = state
        if self.args.rni != 0:
            compute_state = augment_state(state, self.args.rni_num)

        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()

        for t in range(self.args.max_steps):
            misc = dict()
            if self.args.env_graph:
                info['env_graph'] = env_graph
            else:
                info['env_graph'] = self.complete_env_graph

            x = compute_state
            q = self.model(x, info)[0]

            action = []
            for i in range(self.args.nagents):
                if np.random.rand() < self.epsilon:
                    a = np.random.randint(self.model.num_actions)
                else:
                    a = q[i].argmax().item()
                action.append(a)

            action_formatted = [np.array(action)]  # Conform to way env expects the actions

            if not self.epsilon == 0 and use_imitation(args=self.args):  # Use experience (ensure not evaluating)
                next_state, reward, done, info = random.choice(self.experience)
            else:  # Take action using policy
                next_state, reward, done, info = self.env.step(action_formatted)

            next_env_graph = info["env_graph"]

            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            if should_display:
                self.env.display()

            # Add the transition to the buffer
            self.buff.add(state, action, reward, next_state, env_graph, next_env_graph, done)

            state = next_state
            env_graph = next_env_graph

            # Possibly add RNI to state
            compute_state = state
            if self.args.rni != 0:
                compute_state = augment_state(state, self.args.rni_num)

            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            # episode[-1] = episode[-1]._replace(reward=episode[-1].reward + reward)
            # -> Change the reward in the buffer instead of the episode
            experience = list(self.buff.buffer[-1])
            experience[2] += reward
            self.buff.buffer[-1] = tuple(experience)

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return episode, stat

    def compute_grad(self,
                     ):
        stat = dict()

        batch_size = self.args.dgn_batch_size
        batch = self.buff.getBatch(batch_size)
        for j in range(batch_size):
            sample = batch[j]
            if self.args.rni != 0:
                self.O[j] = augment_state(sample[0], self.args.rni_num)
                self.Next_O[j] = augment_state(sample[3], self.args.rni_num)
            else:
                self.O[j] = sample[0]
                self.Next_O[j] = sample[3]
            self.Matrix[j] = sample[4]
            self.Next_Matrix[j] = sample[5]

        q_values = self.model(torch.Tensor(self.O), {"env_graph": torch.Tensor(self.Matrix)})
        target_q_values = self.model_tar(torch.Tensor(self.Next_O), {"env_graph": torch.Tensor(self.Next_Matrix)}).max(dim=2)[0]
        target_q_values = np.array(target_q_values.data)
        expected_q = np.array(q_values.data)

        for j in range(batch_size):
            sample = batch[j]
            for i in range(self.args.nagents):
                expected_q[j][i][sample[1][i]] = sample[2][i] + (1 - sample[6]) * self.args.gamma * target_q_values[j][i]

        loss = (q_values - torch.Tensor(expected_q)).pow(2).mean()
        stat['value_loss'] = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return stat

    def run_batch(self,
                  epoch
                  ):
        # Adds one episode to the buffer
        self.stats = dict()
        self.stats['num_episodes'] = 0
        self.stats['num_steps'] = 0

        episode, episode_stat = self.get_episode(epoch)
        merge_stat(episode_stat, self.stats)
        self.stats['num_episodes'] += 1
        self.stats['num_steps'] += len(episode)

        return self.stats

    # only used when nprocesses=1
    def train_batch(self,
                    epoch
                    ):
        # Works slightly differently to the other trainer
        # 1. Add an episode to the buffer
        # 2. Only train if buffer has enough contents to sample from
        # 3. Update model to target every <args.update_interval> episodes

        # Thus:
        # 1. If empty buffer, add 100 episodes to the buffer
        # 2. Add <args.update_interval> episodes, training <args.train_steps> each time
        # 3. Update the model to the target

        # 1. If empty buffer, add 100 episodes to the buffer
        if self.buff.num_experiences == 0:
            for _ in range(100):
                self.run_batch(epoch)

        # 2. Add <args.update_interval> episodes, training <args.train_steps> each time
        for i in range(self.args.update_interval):
            self.run_batch(epoch)
            for _ in range(self.args.train_steps):
                self.compute_grad()

        # 3. Update the model to the target
        self.model_tar.load_state_dict(self.model.state_dict())

        # Perform evaluation
        stat = {}  # Only report eval stats -> this means that number of episodes won't show up correctly
        num_evals = self.args.num_evals
        epsilon_save = self.epsilon
        self.epsilon = 0
        for i in range(num_evals):
            if i == num_evals - 1:
                self.last_step = True
            s = self.run_batch(epoch)
            merge_stat(s, stat)

        self.epsilon = epsilon_save
        self.last_step = False

        stat["epsilon"] = self.epsilon

        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self,
                        state
                        ):
        self.optimizer.load_state_dict(state)
