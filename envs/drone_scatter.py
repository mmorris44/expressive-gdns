#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drone scatter environment.

Any edge grid cell is a fence.
"""
import copy
import math
import curses
import random

import gym
import numpy as np
from gym import spaces


class _DronePos:
    """
    Wrapper class for drone position, to be able to use np.choice()
    """
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def to_tuple(self):
        return self.w, self.h


class DroneScatterEnv(gym.Env):

    def __init__(self,):
        self.__version__ = "0.0.1"

        # Environment parameters defined in environment args

        self.episode_over = False  # Necessary for returning from step()

    def init_curses(self):
        self.COL_SPACING = 3
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)  # Fence
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # Target
        curses.init_pair(3, curses.COLOR_BLUE, -1)  # Drones

    def init_args(self, parser):
        env = parser.add_argument_group('Drone scattering task')
        env.add_argument('--dim', type=int, default=20,
                         help="Dimension of field area (i.e. side length)")
        env.add_argument('--difficulty', type=str, default='easy', choices={"easy", "hard"},
                         help="Difficulty level. Easy means reward given for being further apart")
        env.add_argument('--comm_range', type=float, default=10,
                         help="Agent communication range")
        env.add_argument('--find_range', type=float, default=3,
                         help="Agent distance to target to count as find")

        # Other environment parameters (usually fixed)
        env.add_argument('--reward_per_time', type=float, default=-1,
                         help="Reward given every time step")
        env.add_argument('--reward_find', type=float, default=100,
                         help="Reward given when the target is found")
        env.add_argument('--reward_per_pairwise_distance', type=float, default=0.01,
                         help="Reward given per average pairwise distance unit")
        env.add_argument('--spawn_density', type=float, default=0.3,
                         help="How much of spawn area should have agents within (spawn area radius will round up)")
        env.add_argument('--min_target_distance', type=float, default=3,
                         help="Minimum distance target should be from spawn area")

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        params = ['dim', 'difficulty', 'comm_range', 'find_range', 'reward_per_time', 'reward_find',
                  'reward_per_pairwise_distance', 'spawn_density', 'min_target_distance']

        for key in params:
            setattr(self, key, getattr(args, key))

        # Initialize environment
        self.ndrones = args.nagents
        self.dims = (self.dim, self.dim)
        self.drone_positions = np.zeros(shape=(self.ndrones, 2), dtype=int)
        self.target_position = np.zeros(shape=2, dtype=int)

        # Actions: 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        self.naction = 4
        self.action_space = spaces.Discrete(self.naction)

        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.naction),  # Last action taken
            spaces.Discrete(2)  # Field or fence
        ))

        self._make_spawn_areas()

        return

    def _make_spawn_areas(self):
        width, height = self.dims

        spawn_width = math.sqrt(self.ndrones / self.spawn_density)  # (spawn width)**2 * density = number of agents
        spawn_radius = math.ceil(spawn_width / 2)

        center_w = width // 2
        center_h = height // 2

        # Agent spawn area
        # (min w, max w) (min h, max h) - min inclusive, max exclusive
        self.drone_spawn_area = ((center_w - spawn_radius, center_w + spawn_radius + 1),
                                 (center_h - spawn_radius, center_h + spawn_radius + 1))
        # Ensure spawn area leaves space for target spawn
        assert self.drone_spawn_area[0][0] > 1 and self.drone_spawn_area[0][1] < width - 1 \
               and self.drone_spawn_area[1][0] > 1 and self.drone_spawn_area[1][1] < height - 1

        # Agent spawn locations
        self.drone_spawn_locations = []
        for w in range(self.drone_spawn_area[0][0], self.drone_spawn_area[0][1]):
            for h in range(self.drone_spawn_area[1][0], self.drone_spawn_area[1][1]):
                self.drone_spawn_locations.append(_DronePos(w, h))

        # Target spawn area
        # (min w, max w) (min h, max h) - min inclusive, max exclusive
        left_target_spawn_area = ((1, self.drone_spawn_area[0][0] - self.min_target_distance + 1),
                                  (1, height - 1))
        right_target_spawn_area = ((self.drone_spawn_area[0][1] + self.min_target_distance, width - 1),
                                   (1, height - 1))
        upper_target_spawn_area = ((left_target_spawn_area[0][1] + 1, right_target_spawn_area[0][0]),
                                   (1, self.drone_spawn_area[1][0] - self.min_target_distance + 1))
        lower_target_spawn_area = ((left_target_spawn_area[0][1] + 1, right_target_spawn_area[0][0]),
                                   (self.drone_spawn_area[1][1] + self.min_target_distance, height - 1))

        self.target_spawn_areas = [left_target_spawn_area, right_target_spawn_area,
                                   upper_target_spawn_area, lower_target_spawn_area]
        # Ensure spawn area has space
        for target_spawn_area in self.target_spawn_areas:
            assert target_spawn_area[0][1] - target_spawn_area[0][0] > 0 \
                   and target_spawn_area[1][1] - target_spawn_area[1][0] > 0

        # Invert spawn areas to have format (min w, min h) (max w, max h)
        for i, target_spawn_area in enumerate(self.target_spawn_areas):
            self.target_spawn_areas[i] = ((target_spawn_area[0][0], target_spawn_area[1][0]),
                                          (target_spawn_area[0][1], target_spawn_area[1][1]))
        self.target_spawn_areas = tuple(self.target_spawn_areas)

    def reset(self, epoch=None):
        self.episode_over = False
        self._spawn_drones()
        self._spawn_target()

        self.last_act = np.zeros(self.ndrones, dtype=int)  # last act STAY when awake

        self.stat = dict()  # Clear stats

        return self._get_obs(), self._get_env_graph()

    def _spawn_drones(self):
        # Set new drone positions in spawn area
        self.drone_positions = np.random.choice(self.drone_spawn_locations, size=self.ndrones, replace=False)
        self.drone_positions = np.array([pos.to_tuple() for pos in self.drone_positions])

    def _spawn_target(self):
        target_spawn_area = random.choice(self.target_spawn_areas)
        self.target_position = np.random.randint(low=target_spawn_area[0], high=target_spawn_area[1])

    def _get_env_graph(self):
        adj = np.zeros((1, self.ndrones, self.ndrones))  # 1 if agents can communicate
        for i in range(self.ndrones):
            for j in range(i + 1, self.ndrones):
                drone_loc1 = self.drone_positions[i]
                drone_loc2 = self.drone_positions[j]
                squared_distance = (drone_loc1[0] - drone_loc2[0]) ** 2 + (drone_loc1[1] - drone_loc2[1]) ** 2
                if squared_distance <= self.comm_range ** 2:
                    adj[0][i][j] = 1
                    adj[0][j][i] = 1

        return adj

    def step(self, action):
        if self.episode_over:
            raise RuntimeError("Episode is done")

        action = np.array(action).squeeze()

        # # For debugging, random actions
        # action = np.random.randint(0, 4, self.ndrones)
        #
        # # For debugging, split up
        # action = np.array([i % 4 for i in range(self.ndrones)])
        #
        # # For debugging, all go one way
        # action = np.array([0] * self.ndrones)

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."
        assert len(action) == self.ndrones, "Action for each agent should be provided."

        for i, a in enumerate(action):
            self._take_action(i, a)

        obs = self._get_obs()
        env_graph = self._get_env_graph()
        reward = self._get_reward()

        self.episode_over = self._target_found()

        extras = {'env_graph': env_graph}

        # Set stats
        # Average pairwise distance between agents
        self.stat['pairwise_distance'] = self._get_average_pairwise_distance()

        return obs, reward, self.episode_over, extras

    def _take_action(self, idx, action):
        # Actions: 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        new_position = np.copy(self.drone_positions[idx])
        width, height = self.dims
        if action == 0:
            new_position[1] += 1
        elif action == 1:
            new_position[0] += 1
        elif action == 2:
            new_position[1] -= 1
        elif action == 3:
            new_position[0] -= 1
        else:
            raise Exception("Invalid action provided:", action)

        # Check if new position within grid
        if new_position[0] < 0 or new_position[0] > width - 1 or new_position[1] < 0 or new_position[1] > height - 1:
            self.last_act[idx] = action  # Drone cannot move outside of field
        else:
            self.last_act[idx] = action
            self.drone_positions[idx] = new_position

    def _get_reward(self):
        reward = np.full(self.ndrones, self.reward_per_time, dtype=float)

        if self.difficulty == "easy":
            for idx in range(self.ndrones):
                reward[idx] += self._get_average_distance(idx) * self.reward_per_pairwise_distance

        if self._target_found():
            reward = np.add(reward, np.full(self.ndrones, self.reward_find))  # All drones rewarded when any find target

        return reward

    def _get_average_distance(self, idx):
        total = 0
        own_pos = self.drone_positions[idx]
        for i, pos in enumerate(self.drone_positions):
            if i == idx:
                continue
            total += math.sqrt((own_pos[0] - pos[0])**2 + (own_pos[1] - pos[1])**2)
        return total / (self.ndrones - 1)

    def _get_average_pairwise_distance(self):
        total = 0
        number = 0
        for i in range(self.ndrones):
            for j in range(i + 1, self.ndrones):
                total += math.sqrt((self.drone_positions[i][0] - self.drone_positions[j][0])**2
                                   + (self.drone_positions[i][1] - self.drone_positions[j][1])**2)
                number += 1
        return total / number

    def _target_found(self):
        for i, pos in enumerate(self.drone_positions):
            if math.sqrt((pos[0] - self.target_position[0])**2 + (pos[1] - self.target_position[1])**2) < self.find_range:
                return True
        return False

    def render(self, mode='human', close=False):
        width, height = self.dims

        self.stdscr.clear()

        # Draw fence
        for w in (0, width - 1):
            for h in range(0, height):
                self.stdscr.addstr(h, w * self.COL_SPACING, "+", curses.color_pair(1))
        for h in (0, height - 1):
            for w in range(1, width - 1):
                self.stdscr.addstr(h, w * self.COL_SPACING, "+", curses.color_pair(1))

        # Draw target
        self.stdscr.addstr(self.target_position[1], self.target_position[0] * self.COL_SPACING,
                           "x", curses.color_pair(2))

        # Draw drones
        for position in self.drone_positions:
            self.stdscr.addstr(position[1], position[0] * self.COL_SPACING,
                               "o", curses.color_pair(3))

        self.stdscr.addstr(width, 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()

    def seed(self, **kwargs):
        return

    def _get_obs(self):
        obs = []
        for i in range(self.ndrones):
            width, height = self.dims
            w, h = self.drone_positions[i]

            # Drone last action
            last_action = self.last_act[i] / (self.naction - 1)

            # What drone sees below
            # 1 denotes the fence around the edge, 0 denotes the field
            below = 1 if (w == 0 or w == width - 1 or h == 0 or h == height - 1) else 0

            o = tuple((last_action, below))
            obs.append(o)

        obs = tuple(obs)
        return obs
