#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Box pushing environment
"""
import copy
import curses
import random

import gym
import numpy as np
from gym import spaces


class _RobotPos:
    """
    Wrapper class for robot position, to be able to use np.choice()
    """
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def to_tuple(self):
        return self.w, self.h


class BoxPushingEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # Environment grid / observation ids
        self._NOTHING = 0  # For when agent is attached
        self._OUTSIDE = 1
        self._EMPTY = 2
        self._BOX = 3
        self._ATTACH_POINT = 4
        self._ROBOT_FREE = 5
        self._ROBOT_ATTACHED = 6
        self._GOAL = 7

        # Number of things agent could see
        self._VOCAB_SIZE = 8

        # Robot modes
        self._FREE_MODE = 0
        self._ATTACHED_MODE = 1

        # Reward parameters
        # self.reward_per_box_time = -0.01
        # self.reward_per_box_exit = 200  # Twice this for a large box
        # self.reward_per_exertion = -1  # When exerting, box does not move
        # self.reward_per_spill = -2  # When spilling, agents know box is tipping and revert to their previous positions
        # self.reward_per_box_move_away = 100  # Reward every time box moves further from its spawn position
        # Negative of the directly above reward given when box moves closer back towards its spawn position

        # Energy used for different actions
        self.ENERGY_PER_DRIVE = 1
        self.ENERGY_PER_POWER_DRIVE = 2
        self.ENERGY_PER_FAILED_DRIVE = 4
        self.ENERGY_PER_FAILED_POWER_DRIVE = 8

        self.episode_over = False  # Necessary for returning from step()
        self._manual_boxes = 0  # Manually spawn in a box type? (0 for no, 1 for large, 2 for small)

    def init_curses(self):
        self.COL_SPACING = 3
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()

        curses_map = {self._BOX: (curses.COLOR_BLUE, "#"),
                      self._ATTACH_POINT: (curses.COLOR_RED, "<"),
                      self._ROBOT_FREE: (curses.COLOR_GREEN, "x"),
                      self._ROBOT_ATTACHED: (curses.COLOR_CYAN, "o"),
                      self._GOAL: (curses.COLOR_YELLOW, "@")}
        self.curses_char_map = {key: char for key, (color, char) in curses_map.items()}

        for key, (color, char) in curses_map.items():
            curses.init_pair(key, color, -1)

    def init_args(self, parser):
        env = parser.add_argument_group('Box pushing task')
        env.add_argument('--dim', type=int, default=5,
                         help="Dimension of construction area (i.e. side length)")
        env.add_argument('--vision', type=int, default=1,
                         help="Vision of robot")
        env.add_argument('--difficulty', type=str, default='easy', choices={"easy", "hard"},
                         help="Difficulty level. Easy means all robots start already attached to the boxes")

        # Environment reward parameters (usually fixed)
        env.add_argument('--reward_per_box_time', type=float, default=-0.01,
                         help="Reward given every time step for each box in the env")
        env.add_argument('--reward_per_box_exit', type=float, default=200,
                         help="Reward given when box leaves the env")
        env.add_argument('--reward_per_exertion', type=float, default=-1,
                         help="Reward given when robot exerts itself")
        env.add_argument('--reward_per_spill', type=float, default=-2,
                         help="Reward given when robot pushes too hard and spills")
        env.add_argument('--reward_per_box_move_away', type=float, default=100,
                         help="Reward given when a box moves further from its spawn position")

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        params = ['dim', 'vision', 'difficulty', 'reward_per_box_time', 'reward_per_box_exit', 'reward_per_exertion',
                  'reward_per_spill', 'reward_per_box_move_away']

        for key in params:
            setattr(self, key, getattr(args, key))

        self.nrobots = args.nagents
        assert self.nrobots >= 8  # Otherwise environment might not be solvable
        self.dims = (self.dim, self.dim)
        assert self.dim >= 12  # Otherwise might not be enough space for spawning
        assert (self.dim ** 2) / 2 > self.nrobots  # Enough space to spawn in robots

        self.small_box_positions = None
        self.large_box_position = None  # Specifies top left of 2x2 box

        self.robots = []  # List of type (position, mode)

        self._make_grid()
        self._make_spawn_areas()
        self._make_action_specs()
        self._make_attached_vis()

        # Actions: 0: STAY, 1: UP, 2: RIGHT, 3: DOWN, 4: LEFT, 5: UP_POW, 6: RIGHT_POW, 7: DOWN_POW, 8: LEFT_POW
        self.naction = 9
        self.action_space = spaces.Discrete(self.naction)

        self.observation_space = spaces.Tuple((
            spaces.Discrete(2),  # Robot mode
            spaces.MultiBinary((2 * self.vision + 1, 2 * self.vision + 1, self._VOCAB_SIZE)),
        ))

        return

    def _make_attached_vis(self):
        # Create what agent sees when attached
        self.attached_vis = np.zeros(shape=(2 * self.vision + 1, 2 * self.vision + 1, self._VOCAB_SIZE))
        self.attached_vis[:, :, self._NOTHING] = 1

    def _make_action_specs(self):
        # Action specs: [num actions, (w change, h change, energy_success, energy_fail)]
        # Actions: 0: STAY, 1: UP, 2: RIGHT, 3: DOWN, 4: LEFT, 5: UP_POW, 6: RIGHT_POW, 7: DOWN_POW, 8: LEFT_POW
        self.action_specs = [[0, 0, 0, 0],
                             [0, -1, self.ENERGY_PER_DRIVE, self.ENERGY_PER_FAILED_DRIVE],
                             [1, 0, self.ENERGY_PER_DRIVE, self.ENERGY_PER_FAILED_DRIVE],
                             [0, 1, self.ENERGY_PER_DRIVE, self.ENERGY_PER_FAILED_DRIVE],
                             [-1, 0, self.ENERGY_PER_DRIVE, self.ENERGY_PER_FAILED_DRIVE],
                             [0, -1, self.ENERGY_PER_POWER_DRIVE, self.ENERGY_PER_FAILED_POWER_DRIVE],
                             [1, 0, self.ENERGY_PER_POWER_DRIVE, self.ENERGY_PER_FAILED_POWER_DRIVE],
                             [0, 1, self.ENERGY_PER_POWER_DRIVE, self.ENERGY_PER_FAILED_POWER_DRIVE],
                             [-1, 0, self.ENERGY_PER_POWER_DRIVE, self.ENERGY_PER_FAILED_POWER_DRIVE]]
        self.action_specs = np.array(self.action_specs)

    def _make_grid(self):
        # Grid indexed from top left
        self.grid = np.full(self.dims[0] * self.dims[1], self._EMPTY, dtype=int).reshape(self.dims)

    def _make_spawn_areas(self):
        # Agents can spawn at random within the goal area

        self._set_grid()  # To read where goal locations are

        width, height = self.dims
        # Agent spawn locations
        self.robot_spawn_locations = []
        for w in range(width):
            for h in range(height):
                if self.grid[w, h] == self._GOAL:
                    self.robot_spawn_locations.append(_RobotPos(w, h))

        # Box spawn locations
        center_w = width // 2
        center_h = height // 2
        self.large_box_spawn = (center_w - 1, center_h - 1)
        self.small_box_spawn = [(center_w - 2, center_h), (center_w + 1, center_h)]

        # Robot attached spawn locations
        # Simulate spawning and create grid to generate points
        self.large_robot_attached_spawn_locations = []
        self._spawn_large_box()
        self._set_grid()
        for w in range(width):
            for h in range(height):
                if self.grid[w, h] == self._ATTACH_POINT:
                    self.large_robot_attached_spawn_locations.append((w, h))

        self.small_robot_attached_spawn_locations = []
        self._spawn_small_boxes()
        self._set_grid()
        for w in range(width):
            for h in range(height):
                if self.grid[w, h] == self._ATTACH_POINT:
                    self.small_robot_attached_spawn_locations.append((w, h))

    def _set_grid(self):
        # Clear grid
        self.grid[:, :] = self._EMPTY
        width, height = self.dims

        # Set outmost 3 rows / cols to be goal areas
        self.grid[:3, :] = self._GOAL
        self.grid[width - 3:, :] = self._GOAL
        self.grid[:, :3] = self._GOAL
        self.grid[:, height - 3:] = self._GOAL

        # Set boxes and attach points
        if self.small_box_positions is not None:
            for (w, h) in self.small_box_positions:
                self.grid[w, h] = self._BOX
                self.grid[w - 1, h] = self._ATTACH_POINT
                self.grid[w + 1, h] = self._ATTACH_POINT
                self.grid[w, h - 1] = self._ATTACH_POINT
                self.grid[w, h + 1] = self._ATTACH_POINT

        if self.large_box_position is not None:
            w, h = self.large_box_position
            self.grid[w:w+2, h:h+2] = self._BOX
            self.grid[w-1, h:h+2] = self._ATTACH_POINT
            self.grid[w+2, h:h+2] = self._ATTACH_POINT
            self.grid[w:w+2, h-1] = self._ATTACH_POINT
            self.grid[w:w+2, h+2] = self._ATTACH_POINT

        # Set agents
        for (position, mode) in self.robots:
            if mode == self._FREE_MODE:
                self.grid[position] = self._ROBOT_FREE
            elif mode == self._ATTACHED_MODE:
                self.grid[position] = self._ROBOT_ATTACHED
            else:
                raise Exception("Unrecognized robot mode:", mode)

    def _position_in_goal_area(self, position):
        w, h = position
        width, height = self.dims
        return w <= 2 or w >= width - 3 or h <= 2 or h >= height - 3

    def reset(self, epoch=None):
        self.episode_over = False
        self.energy_used = 0
        self.num_exertions = 0
        self.num_spills = 0
        self.small_boxes_cleared = 0
        self.large_boxes_cleared = 0

        self.stat = dict()  # Clear stats

        self._spawn_boxes()
        if self.difficulty == 'easy':
            self._spawn_robots_easy()
        else:
            raise Exception("Unsupported difficulty:", self.difficulty)

        self._set_grid()

        return self._get_obs(), self._get_env_graph()

    def _spawn_boxes(self):
        if self._manual_boxes == 0:
            if random.random() < 0.5:
                self._spawn_small_boxes()
            else:
                self._spawn_large_box()
        else:  # Manually spawn in a box type (0 for no, 1 for large, 2 for small)
            if self._manual_boxes == 1:
                self._spawn_large_box()
            elif self._manual_boxes == 2:
                self._spawn_small_boxes()
            else:
                raise Exception("Unexpected manual box value:", self._manual_boxes)

    def _spawn_small_boxes(self):
        # Spawn two small boxes
        self.large_box_position = None
        self.small_box_positions = copy.deepcopy(self.small_box_spawn)

    def _spawn_large_box(self):
        # Spawn one large box
        self.small_box_positions = None
        self.large_box_position = copy.deepcopy(self.large_box_spawn)

    def _spawn_robots_easy(self):
        # Spawn 8 already attached and the rest at random
        self.robots = [None] * self.nrobots
        agent_ids = list(range(self.nrobots))
        random.shuffle(agent_ids)
        num_allocated = 0

        if self.small_box_positions is None:
            attached_spawn_locations = self.large_robot_attached_spawn_locations
        else:
            attached_spawn_locations = self.small_robot_attached_spawn_locations

        # Spawn around boxes
        for i, agent_id in enumerate(agent_ids):
            if num_allocated == 8:
                break

            self.robots[agent_id] = [attached_spawn_locations[i], self._ATTACHED_MODE]
            num_allocated += 1

        # Spawn remainder at random
        self._spawn_robots_at_random(agent_ids[8:])

    def _spawn_robots_at_random(self, agent_ids):
        # Set new robot positions in random spawn area
        positions = np.random.choice(self.robot_spawn_locations, size=len(agent_ids), replace=False)

        for i, agent_id in enumerate(agent_ids):
            self.robots[agent_id] = [positions[i].to_tuple(), self._FREE_MODE]

    def _get_env_graph(self):
        adj = np.zeros((1, self.nrobots, self.nrobots))  # 1 if agents can communicate

        # Get free agents
        free_agent_ids = []
        for agent_id, (position, mode) in enumerate(self.robots):
            if mode == self._FREE_MODE:
                free_agent_ids.append(agent_id)

        # Free agents can all communicate with each other
        for i, agent_id in enumerate(free_agent_ids):
            adj[0][agent_id][free_agent_ids] = 1
            adj[0][agent_id][agent_id] = 0  # Agent cannot communicate with itself

        # Get group of sequences of agent positions
        agent_positions_group = []
        if self.large_box_position is not None:  # Add sequence of agent positions around large box
            agent_positions_group = [self._get_positions_adjacent_to_large_box(self.large_box_position)]
        elif self.small_box_positions is not None:  # Add sequence of agent positions around small boxes
            for small_box_position in self.small_box_positions:
                agent_positions_group.append(self._get_positions_adjacent_to_small_box(small_box_position))

        # Connect adjacent agents
        for agent_positions in agent_positions_group:
            agent_positions.append(agent_positions[0])  # To complete the circuit
            # Get ids of agents in the positions
            agent_ids = self._get_robot_ids_in_positions_respect_order(agent_positions)

            # Connect agents in both communication directions
            for i in range(len(agent_ids) - 1):
                adj[0][agent_ids[i]][agent_ids[i + 1]] = 1
                adj[0][agent_ids[i + 1]][agent_ids[i]] = 1

        return adj

    def step(self, actions):
        if self.episode_over:
            raise RuntimeError("Episode is done")

        actions = np.array(actions).squeeze()
        reward = self._get_step_reward()  # All agents given negative reward for boxes

        assert np.all(actions <= self.naction), "Actions should be in the range [0,naction)."
        assert len(actions) == self.nrobots, "Action for each agent should be provided."

        # # For debugging, manually override the actions to move right cleverly
        # if self.small_box_positions is not None:
        #     actions = np.full(self.nrobots, 1)
        # else:
        #     actions = np.full(self.nrobots, 5)

        # # For debugging, always move right with pow actions
        # actions = np.full(self.nrobots, 5)

        # # For debugging, random actions
        # actions = np.random.randint(0, 9, self.nrobots)

        r = self._take_actions(actions)
        reward = np.add(reward, r)

        r = self._remove_cleared_boxes()
        reward = np.add(reward, r)
        self.episode_over = self._episode_over()

        obs = self._get_obs()
        env_graph = self._get_env_graph()

        extras = {'env_graph': env_graph}

        self.stat['energy_used'] = self.energy_used
        self.stat['num_exertions'] = self.num_exertions
        self.stat['num_spills'] = self.num_spills
        self.stat['ratio_boxes_cleared'] = (self.small_boxes_cleared + 2 * self.large_boxes_cleared) / 2
        self.stat['small_boxes_cleared'] = self.small_boxes_cleared
        self.stat['large_boxes_cleared'] = self.large_boxes_cleared

        return obs, reward, self.episode_over, extras

    def _take_actions(self, actions):
        # Action specs: [num actions, (w change, h change, energy_success, energy_fail)]
        # Actions: 0: STAY, 1: UP, 2: RIGHT, 3: DOWN, 4: LEFT, 5: UP_POW, 6: RIGHT_POW, 7: DOWN_POW, 8: LEFT_POW
        reward = np.zeros(self.nrobots)

        # Take free robot actions first
        for i, (position, mode) in enumerate(self.robots):
            if mode == self._FREE_MODE:
                self._take_action_free(i, actions[i])

        # Take attached robot actions
        if self.large_box_position is not None:
            position = self.large_box_position
            adjacent_positions = self._get_positions_adjacent_to_large_box(position)
            agent_ids = self._get_robot_ids_in_positions(adjacent_positions)  # Collect list of attached robots

            r = self._take_actions_attached(agent_ids, actions[agent_ids], on_large_box=True)
            reward = np.add(reward, r)
        elif self.small_box_positions is not None:
            for box_index, position in enumerate(self.small_box_positions):
                adjacent_positions = self._get_positions_adjacent_to_small_box(position)
                agent_ids = self._get_robot_ids_in_positions(adjacent_positions)  # Collect list of attached robots

                r = self._take_actions_attached(agent_ids, actions[agent_ids], small_box_index=box_index)
                reward = np.add(reward, r)

        return reward

    def _take_actions_attached(self, agent_ids, actions, on_large_box=False, small_box_index=None):
        # All agents are attached to the same box when this function is called
        reward = np.zeros(self.nrobots)
        action = actions[0]

        # If agents are not all taking the same action
        if not np.all(actions == action):
            for index, action in enumerate(actions):
                agent_id = agent_ids[index]

                # Give negative reward for exertion
                if action != 0:
                    reward[agent_id] = self.reward_per_exertion

                # Update energy
                self.energy_used += self.action_specs[action][3]
        elif action == 0:
            pass  # Do nothing for all STAY action
        else:  # Agents all taking 'action' now
            # If wrong action for box type, either exert or spill
            if 1 <= action <= 4 and on_large_box:  # Exert on large box
                reward[agent_ids] = self.reward_per_exertion
                self.energy_used += len(agent_ids) * self.action_specs[action][3]
                self.num_exertions += 1
            elif 5 <= action <= 8 and not on_large_box:  # Spill on small box
                reward[agent_ids] = self.reward_per_spill
                self.energy_used += len(agent_ids) * self.action_specs[action][2]  # Move uses 'successful' energy
                self.num_spills += 1
            else:
                reward = self._take_approved_action_attached(agent_ids, action, on_large_box, small_box_index)

        return reward

    def _take_approved_action_attached(self, agent_ids, action, on_large_box, small_box_index):
        # All agents attached to same box, taking correct action for the box when this is called
        reward = np.zeros(self.nrobots)
        action_spec = self.action_specs[action]  # (w change, h change, energy_success, energy_fail)

        # Generate list of new positions, clear grid of box and involved agents
        # If any new position is filled, the move fails -> apply exertion reward and energy usage
        # Then put box and agents back into grid

        # Generate list of new positions
        agent_positions = []
        new_agent_positions = []
        for agent_id, (position, mode) in enumerate(self.robots):
            if agent_id in agent_ids:
                new_position = (position[0] + action_spec[0], position[1] + action_spec[1])
                new_agent_positions.append(new_position)
                agent_positions.append(position)

        if on_large_box:
            box_locations = [(self.large_box_position[0],
                              self.large_box_position[1]),

                             (self.large_box_position[0] + 1,
                              self.large_box_position[1]),

                             (self.large_box_position[0],
                              self.large_box_position[1] + 1),

                             (self.large_box_position[0] + 1,
                              self.large_box_position[1] + 1)]

            new_box_locations = [(self.large_box_position[0] + action_spec[0],
                                  self.large_box_position[1] + action_spec[1]),

                                 (self.large_box_position[0] + action_spec[0] + 1,
                                  self.large_box_position[1] + action_spec[1]),

                                 (self.large_box_position[0] + action_spec[0],
                                  self.large_box_position[1] + action_spec[1] + 1),

                                 (self.large_box_position[0] + action_spec[0] + 1,
                                  self.large_box_position[1] + action_spec[1] + 1)]
        else:
            box_locations = [self.small_box_positions[small_box_index]]

            new_box_locations = [(self.small_box_positions[small_box_index][0] + action_spec[0],
                                  self.small_box_positions[small_box_index][1] + action_spec[1])]

        # Clear grid of box
        for position in box_locations:
            self._clear_grid_cell(position)

        # Clear grid of involved agents
        for position in agent_positions:
            self._clear_grid_cell(position)

        # If any new position is filled, the move fails (only needed to check agent positions)
        legal_move = True
        for position in new_agent_positions:
            if self.grid[position] not in {self._EMPTY, self._GOAL}:
                legal_move = False
                break

        if not legal_move:
            # Apply exertion reward and energy usage
            reward[agent_ids] = self.reward_per_exertion
            self.energy_used += action_spec[3] * len(agent_ids)
            self.num_exertions += 1

            # Then put box and agents back into grid
            for agent_position in agent_positions:
                self.grid[agent_position] = self._ROBOT_ATTACHED
            if on_large_box:
                w, h = self.large_box_position
                self.grid[w:w + 2, h:h + 2] = self._BOX
            else:
                w, h = self.small_box_positions[small_box_index]
                self.grid[w, h] = self._BOX

        # If all new positions are available, move is legal
        # Give reward if box moved closer to the edge
        # Update agent positions, update box position, update grid, update energy usage
        else:
            # Give reward if box moved further from its spawn position
            if on_large_box:
                box_spawn = self.large_box_spawn
                box_prev_pos = self.large_box_position
            else:
                box_spawn = self.small_box_spawn[small_box_index]
                box_prev_pos = self.small_box_positions[small_box_index]
            box_new_pos = new_box_locations[0]

            if (box_new_pos[0] - box_spawn[0]) ** 2 + (box_new_pos[1] - box_spawn[1]) ** 2\
                    > (box_prev_pos[0] - box_spawn[0]) ** 2 + (box_prev_pos[1] - box_spawn[1]) ** 2:
                reward[agent_ids] = self.reward_per_box_move_away
            else:  # Otherwise give negative reward (since box moved closer towards its spawn position)
                reward[agent_ids] = -self.reward_per_box_move_away

            # Update agent positions
            for index, agent_id in enumerate(agent_ids):
                new_position = new_agent_positions[index]
                self.robots[agent_id] = (new_position, self.robots[agent_id][1])

            # Update box position
            if on_large_box:
                self.large_box_position = new_box_locations[0]
            else:
                self.small_box_positions[small_box_index] = new_box_locations[0]

            # Update grid
            for agent_position in new_agent_positions:
                self.grid[agent_position] = self._ROBOT_ATTACHED
            for box_location in new_box_locations:
                self.grid[box_location] = self._BOX

            # Update energy usage
            self.energy_used += action_spec[2] * len(agent_ids)

        return reward

    def _take_action_free(self, agent_id, action):
        # Actions: 0: STAY, 1: UP, 2: RIGHT, 3: DOWN, 4: LEFT, 5: UP_POW, 6: RIGHT_POW, 7: DOWN_POW, 8: LEFT_POW
        position = self.robots[agent_id][0]
        width, height = self.dims

        new_position = (position[0] + self.action_specs[action][0], position[1] + self.action_specs[action][1])

        # Check if new position within grid
        if new_position[0] < 0 or new_position[0] > width - 1 or new_position[1] < 0 or new_position[1] > height - 1:
            self.energy_used += self.action_specs[action][3]  # Failed energy used
        elif self.grid[new_position] not in {self._EMPTY, self._GOAL}:  # Check if obstacle in new position
            self.energy_used += self.action_specs[action][3]  # Failed energy used
        else:
            # Move is legal: update position, update grid, update energy usage
            self.robots[agent_id] = (new_position, self.robots[agent_id][1])
            self._clear_grid_cell(position)
            self.grid[new_position] = self._ROBOT_FREE
            self.energy_used += self.action_specs[action][2]

    def _get_robot_ids_in_positions(self, positions):
        robot_ids = []

        for agent_id, (position, mode) in enumerate(self.robots):
            if position in positions:
                robot_ids.append(agent_id)

        return np.array(robot_ids)

    def _get_robot_ids_in_positions_respect_order(self, positions):
        # Same as above method, but will provide duplicate agents if there are duplicate positions
        # Also respects the ordering of positions, and will return agent ids in the corresponding order
        robot_ids = []

        for position in positions:
            for agent_id, (agent_position, mode) in enumerate(self.robots):
                if position == agent_position:
                    robot_ids.append(agent_id)
                    break

        return robot_ids

    def _get_positions_adjacent_to_small_box(self, position):
        adjacent_positions = [(position[0], position[1] + 1),
                              (position[0], position[1] - 1),
                              (position[0] + 1, position[1]),
                              (position[0] - 1, position[1])]
        return adjacent_positions

    def _get_positions_adjacent_to_large_box(self, position):
        adjacent_positions = [(position[0] - 1, position[1]),
                              (position[0] - 1, position[1] + 1),
                              (position[0], position[1] - 1),
                              (position[0] + 1, position[1] - 1),
                              (position[0] + 2, position[1]),
                              (position[0] + 2, position[1] + 1),
                              (position[0], position[1] + 2),
                              (position[0] + 1, position[1] + 2)]
        return adjacent_positions

    def _remove_cleared_boxes(self):
        reward = np.zeros(self.nrobots)
        # Make position None (or remove one), free agents, update grid, give reward to each agent that participated
        if self.small_box_positions is not None:
            box_id = 0
            while True:
                position = self.small_box_positions[box_id]
                if self._position_in_goal_area(position):
                    # Make position None (or remove from list)
                    if len(self.small_box_positions) == 1:
                        self.small_box_positions = None
                    else:
                        self.small_box_positions.pop(box_id)

                    adjacent_positions = self._get_positions_adjacent_to_small_box(position)

                    reward = self._remove_cleared_box([position], adjacent_positions)
                else:
                    box_id += 1  # Move to next box if not in goal area

                # Last box reached if list is None or at end of list
                if self.small_box_positions is None or box_id == len(self.small_box_positions):
                    break

        elif self.large_box_position is not None:
            position = self.large_box_position

            # All positions filled by the box
            box_positions = [(position[0], position[1]),
                             (position[0], position[1] + 1),
                             (position[0] + 1, position[1]),
                             (position[0] + 1, position[1] + 1)]
            box_done = True  # Box only done when completely in goal area
            for box_position in box_positions:
                if not self._position_in_goal_area(box_position):
                    box_done = False
                    break

            if box_done:
                # Make position None
                self.large_box_position = None

                adjacent_positions = self._get_positions_adjacent_to_large_box(position)

                reward = self._remove_cleared_box(box_positions, adjacent_positions)

        return reward

    def _remove_cleared_box(self, box_positions, adjacent_positions):
        # Free agents, gives rewards to agents that participated, update the grid, log clearing
        reward = np.zeros(self.nrobots)

        for agent_id, (position, mode) in enumerate(self.robots):
            if position in adjacent_positions:
                # Free agents
                self.robots[agent_id] = (self.robots[agent_id][0], self._FREE_MODE)

                # Give rewards to each agent that participated
                if len(box_positions) > 1:  # Large box
                    reward[agent_id] = 2 * self.reward_per_box_exit
                else:
                    reward[agent_id] = self.reward_per_box_exit

        # Update grid
        for box_position in box_positions:
            self._clear_grid_cell(box_position)
        for robot_position in adjacent_positions:
            self._clear_grid_cell(robot_position)

        # Log clearing
        if len(box_positions) > 1:  # Large box
            self.large_boxes_cleared += 1
        else:
            self.small_boxes_cleared += 1

        return reward

    def _clear_grid_cell(self, position):
        # Sets it back to either EMPTY or GOAL, depending on position
        self.grid[position] = self._GOAL if self._position_in_goal_area(position) else self._EMPTY

    def _episode_over(self):
        # Positions set to None when all boxes have been cleared
        return self.small_box_positions is None and self.large_box_position is None

    def render(self, mode='human', close=False):
        width, height = self.dims

        self.stdscr.clear()

        # Iterate through grid
        for w in range(width):
            for h in range(height):
                key = self.grid[w, h]
                if key in self.curses_char_map:
                    self.stdscr.addstr(h, w * self.COL_SPACING, self.curses_char_map[key], curses.color_pair(key))

        self.stdscr.addstr(width, 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()

    def seed(self):
        return

    def _get_step_reward(self):
        boxes_remaining = 2
        if self.small_box_positions is not None:
            boxes_remaining -= len(self.small_box_positions)
        elif self.large_box_position is not None:
            boxes_remaining -= 1

        return np.full(self.nrobots, boxes_remaining * self.reward_per_box_time, dtype=float)

    def _get_obs(self):
        # self.observation_space = spaces.Tuple((
        #             spaces.Discrete(2),  # Robot mode
        #             spaces.MultiBinary((2 * self.vision + 1, 2 * self.vision + 1, self._VOCAB_SIZE)),
        #         ))
        obs = []

        bool_grid = self._get_bool_grid()

        for agent_id, (position, mode) in enumerate(self.robots):
            mode_obs = mode

            # Cannot see anything if attached
            if mode == self._ATTACHED_MODE:
                vision_obs = self.attached_vis
            else:
                # Get vision slice
                slice_w = slice(position[0], position[0] + (2 * self.vision) + 1)
                slice_h = slice(position[1], position[1] + (2 * self.vision) + 1)
                vision_obs = bool_grid[slice_w, slice_h]

            o = tuple((mode_obs, vision_obs))
            obs.append(o)

        obs = tuple(obs)
        return obs

    def _get_bool_grid(self):
        # Get one-hot encoding of grid
        # Pad every side with self.vision squares of self._OUTSIDE
        width, height = self.dims
        # Nothing flagged initially
        bool_grid = np.full(shape=(width + 2 * self.vision, height + 2 * self.vision, self._VOCAB_SIZE), fill_value=0)

        # Pad sides with self._OUTSIDE
        bool_grid[:self.vision, :, self._OUTSIDE] = 1  # Left
        bool_grid[-self.vision:, :, self._OUTSIDE] = 1  # Right
        bool_grid[:, :self.vision, self._OUTSIDE] = 1  # Top
        bool_grid[:, -self.vision:, self._OUTSIDE] = 1  # Bottom

        # Fill bool grid with grid info
        for w in range(width):
            for h in range(height):
                # Shift indices along by vision
                bool_grid[w + self.vision, h + self.vision, self.grid[w, h]] = 1
        return bool_grid

    def _print_bool_grid(self, bool_grid):
        # For debugging
        for w in range(bool_grid.shape[0]):
            for h in range(bool_grid.shape[1]):
                print(bool_grid[w, h], end=" ")
            print("\n")
        print("\n\n\n")
