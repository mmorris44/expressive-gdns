import sys
import signal
import argparse
import os
import random
from pathlib import Path

import visdom

from src.baselines.comm import CommNetMLP
from src.baselines.ga_comm import GACommNetMLP
from src.baselines.tar_comm import TarCommNetMLP
from src.baselines.DGN import DGN
from src.magic import MAGIC
from src.baselines.trainer import BaselineTrainer
from src.baselines.dgn_trainer import DGNTrainer
from src.trainer import MagicTrainer

import data
from utils import *
from action_utils import parse_action_args
from multi_processing import MultiProcessTrainer


# Register custom environments
register_custom_envs()

# Set number of threads to 1 to help with Pytorch performance issues on Linux
torch.set_num_threads(1)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='Number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=1,
                    help='How many processes to run')
# DGN training
parser.add_argument('--update_interval', type=int, default=5,
                    help='How many episodes between model update steps (for DGN only)')
parser.add_argument('--train_steps', type=int, default=5,
                    help='How many times to train the model in a training step (for DGN only)')
parser.add_argument('--dgn_batch_size', type=int, default=128,
                    help='Batch size (for DGN only)')
parser.add_argument('--epsilon_start', default=1, type=float,
                    help='Epsilon starting value')
parser.add_argument('--epsilon_min', default=0.1, type=float,
                    help='Minimum epsilon value')
parser.add_argument('--epsilon_step', default=0.0004, type=float,
                    help='Amount to subtract from epsilon each episode')
parser.add_argument('--buffer_capacity', default=40000, type=int,
                    help='Capacity of the replay buffer')
# model
parser.add_argument('--model', default="error", type=str, choices={"gacomm", "commnet", "tarmac", "ic3net",
                                                                   "tarmac_ic3net", "dgn", "magic"},
                    help='name of model to use')
parser.add_argument('--hid_size', default=128, type=int,
                    help='hidden layer size')
parser.add_argument('--qk_hid_size', default=16, type=int,
                    help='key and query size for soft attention')
parser.add_argument('--value_hid_size', default=32, type=int,
                    help='value size for soft attention')
parser.add_argument('--recurrent', type=int, default=0,
                    help='make the model recurrent in time')
parser.add_argument('--num_evals', type=int, default=10,
                    help='Number of evaluation runs with each training iteration')

# RNI
parser.add_argument('--rni', default=0, type=float,
                    help='fraction of initial node features to come from RNI. 0 for none, 1 to denote using agent IDs')
# Imitation learning
parser.add_argument('--imitation', type=int, default=0,
                    help='Whether to use imitation learning during training')
parser.add_argument('--num_imitation_experiences', default=100, type=int,
                    help='Number of experiences to come from imitation')
parser.add_argument('--num_normal_experiences', default=900, type=int,
                    help='Number of experiences to come from normal policy')

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed')  # TODO: works in thread?
parser.add_argument('--normalize_rewards', type=int, default=0,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# other
parser.add_argument('--plot', type=int, default=0,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--plot_port', default='8097', type=str,
                    help='plot port')
parser.add_argument('--save', type=int, default=0,
                    help='save the model after training')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', type=int, default=0,
                    help='Display environment state')
parser.add_argument('--random', type=int, default=0,
                    help="enable random model")
parser.add_argument('--use_wandb', type=int, default=0,
                    help="whether to use wandb")
parser.add_argument('--wandb_log_interval', type=int, default=1,
                    help="how often (number of epochs) to send logs to wandb")
parser.add_argument('--greedy_a2c_eval', type=int, default=0,
                    help="whether to evaluate a2c methods greedily instead of stochastically")

# CommNet specific args
parser.add_argument('--commnet', type=int, default=0,
                    help="enable commnet model")
parser.add_argument('--ic3net', type=int, default=0,
                    help="enable ic3net model")
parser.add_argument('--tarcomm', type=int, default=0,
                    help="enable tarmac model (with commnet or ic3net)")
parser.add_argument('--gacomm', type=int, default=0,
                    help="enable gacomm model")
parser.add_argument('--dgn', type=int, default=0,
                    help="enable dgn model")
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=2,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', type=int, default=0,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                         + ' Default 10 (lower than previous)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', type=int, default=0,
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', type=int, default=0,
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', type=int, default=0,
                    help='Whether to multiply log prob for each chosen action with advantages')
parser.add_argument('--share_weights', type=int, default=0,
                    help='Share weights for hops')
parser.add_argument('--env_graph', type=int, default=0,
                    help='Whether to use the communication graph returned by the environment')

# MAGIC specific args
parser.add_argument('--magic', type=int, default=0,
                    help="enable MAGIC model")
parser.add_argument('--directed', type=int, default=0,
                    help='whether the communication graph is directed')
parser.add_argument('--self_loop_type', default=1, type=int,
                    help='self loop type in the gat layers (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--gat_num_heads', default=4, type=int,
                    help='number of heads in gat layers except the last one')
parser.add_argument('--gat_num_heads_out', default=1, type=int,
                    help='number of heads in output gat layer')
parser.add_argument('--gat_hid_size', default=32, type=int,
                    help='hidden size of one head in gat')
parser.add_argument('--message_decoder', type=int, default=0,
                    help='whether use the message decoder')
parser.add_argument('--gat_normalize', type=int, default=0,
                    help='whether normalize the coefficients in the gat layers of the message processor')
parser.add_argument('--ge_num_heads', default=4, type=int,
                    help='number of heads in the gat encoder')
parser.add_argument('--gat_encoder_normalize', type=int, default=0,
                    help='whether normilize the coefficients in the gat encoder (they have been normalized if the input graph is complete)')
parser.add_argument('--use_gat_encoder', type=int, default=0,
                    help='whether use the gat encoder before learning the first graph')
parser.add_argument('--gat_encoder_out_size', default=64, type=int,
                    help='hidden size of output of the gat encoder')
parser.add_argument('--graph_complete', type=int, default=0,
                    help='whether the communication graph is set to a complete graph')
parser.add_argument('--learn_different_graphs', type=int, default=0,
                    help='whether learn a new communication graph at each round of communication')
parser.add_argument('--message_encoder', type=int, default=0,
                    help='whether use the message encoder')


init_args_for_env(parser)
args = parser.parse_args()

# Check if wandb should be imported
if args.use_wandb:
    import wandb

    wandb.init(project="wl-gdn", entity="mmorris44")

# Set model type flags from string
# Option: {"gacomm, commnet, tarmac, ic3net, dgn, magic"}
if args.model == "gacomm":
    args.gacomm = True
elif args.model == "commnet":
    args.commnet = True
    args.tarcomm = False
elif args.model == "tarmac":
    args.commnet = True
    args.tarcomm = True
elif args.model == "ic3net":
    args.ic3net = True
    args.tarcomm = False
elif args.model == "tarmac_ic3net":
    args.ic3net = True
    args.tarcomm = True
elif args.model == "dgn":
    args.dgn = True
elif args.model == "magic":
    args.magic = True
else:
    raise Exception("Model name needs to be specified in arguments")

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games

    # Removed -> want to test actual comms. Same below for gacomm
    # if args.env_name == "traffic_junction":
    #     args.comm_action_one = True

if args.gacomm:
    args.commnet = 1
    args.mean_ratio = 0
    # if args.env_name == "traffic_junction":
    #     args.comm_action_one = True

# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

if args.env_name == 'grf':
    render = args.render
    args.render = False
env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)):  # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'

# Parse action arguments (e.g. continuous)
parse_action_args(args)

# Fix randomness
if args.seed == -1:
    args.seed = np.random.randint(0, 10000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Scale num_inputs for RNI
if args.rni != 0 and args.rni != 1:
    num_inputs = int(env.observation_dim / (1 - args.rni))  # A fraction of <args.rni> inputs are random
    args.rni_num = num_inputs - env.observation_dim

    if args.rni_num == 0:
        raise Exception("Not enough observation dimensions to provide the requested amount of RNI")

# If rni is 1, take this to mean that agent IDs should be concatenated to observations
if args.rni == 1:
    args.rni_num = -1  # This will be used by RNI_utils.augment_state to infer that unique IDs should be used
    num_inputs = env.observation_dim + args.nagents

args.obs_size = num_inputs  # For MAGIC

print(args)

if args.gacomm:  # GA-Comm
    policy_net = GACommNetMLP(args, num_inputs)
elif args.dgn:  # DGN
    policy_net = DGN(args, num_inputs)
elif args.commnet:
    if args.tarcomm:  # TARMAC
        policy_net = TarCommNetMLP(args, num_inputs)
    else:  # CommNet
        policy_net = CommNetMLP(args, num_inputs)
elif args.magic:
    policy_net = MAGIC(args)
else:
    raise Exception("Model not specified in args")

# Watch model if using wandb
if args.use_wandb:
    wandb.watch(policy_net)

if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

if args.env_name == 'grf':
    args.render = render
if args.nprocesses > 1:  # Multi process trainer is broken
    trainer = MultiProcessTrainer(args, lambda: BaselineTrainer(args, policy_net, data.init(args.env_name, args)))
else:
    if args.dgn:
        trainer = DGNTrainer(args, policy_net, data.init(args.env_name, args))
    elif args.magic:
        trainer = MagicTrainer(args, policy_net, data.init(args.env_name, args))
    else:
        trainer = BaselineTrainer(args, policy_net, data.init(args.env_name, args))

# Commented the following out, as it seems unused
# disp_trainer = BaselineTrainer(args, policy_net, data.init(args.env_name, args, False))
# disp_trainer.display = True
# def disp():
#     x = disp_trainer.get_episode()

log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
log['density1'] = LogField(list(), True, 'epoch', 'num_steps')
log['density2'] = LogField(list(), True, 'epoch', 'num_steps')
log['pairwise_distance'] = LogField(list(), True, 'epoch', 'num_episodes')
log['energy_used'] = LogField(list(), True, 'epoch', 'num_episodes')
log['num_exertions'] = LogField(list(), True, 'epoch', 'num_episodes')
log['num_spills'] = LogField(list(), True, 'epoch', 'num_episodes')
log['ratio_boxes_cleared'] = LogField(list(), True, 'epoch', 'num_episodes')
log['small_boxes_cleared'] = LogField(list(), True, 'epoch', 'num_episodes')
log['large_boxes_cleared'] = LogField(list(), True, 'epoch', 'num_episodes')

if args.plot:
    vis = visdom.Visdom(env=args.plot_env, port=args.plot_port)
if args.gacomm:
    model_dir = Path('./saved') / args.env_name / 'gacomm'
elif args.tarcomm:
    if args.ic3net:
        model_dir = Path('./saved') / args.env_name / 'tar_ic3net'
    elif args.commnet:
        model_dir = Path('./saved') / args.env_name / 'tar_commnet'
    else:
        model_dir = Path('./saved') / args.env_name / 'other'
elif args.ic3net:
    model_dir = Path('./saved') / args.env_name / 'ic3net'
elif args.commnet:
    model_dir = Path('./saved') / args.env_name / 'commnet'
elif args.dgn:
    model_dir = Path('./saved') / args.env_name / 'dgn'
else:
    model_dir = Path('./saved') / args.env_name / 'other'
if args.env_name == 'grf':
    model_dir = model_dir / args.scenario
if not model_dir.exists():
    curr_run = 'run1'
else:
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                     model_dir.iterdir() if
                     str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
run_dir = model_dir / curr_run


def run(num_epochs):
    num_episodes = 0
    if args.save:
        os.makedirs(run_dir)
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
            s = trainer.train_batch(ep)
            print('batch: ', n)
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        num_episodes += stat['num_episodes']
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=2)

        wandb_log = {
            'epoch': epoch,
            'episode': num_episodes,
            'total_reward': sum(stat['reward']),
            'time': epoch_time
        }

        print('Epoch {}'.format(epoch))
        print('Episode: {}'.format(num_episodes))
        print('Reward: {}'.format(stat['reward']))
        print('Total Reward: {}'.format(sum(stat['reward'])))
        print('Time: {:.2f}s'.format(epoch_time))

        if 'enemy_reward' in stat.keys():
            print('Enemy-Reward: {}'.format(stat['enemy_reward']))
            wandb_log['total_enemy_reward'] = sum(stat['enemy_reward'])
        if 'add_rate' in stat.keys():
            print('Add-Rate: {:.2f}'.format(stat['add_rate']))
            wandb_log['add_rate'] = stat['add_rate']
        if 'success' in stat.keys():
            print('Success: {:.4f}'.format(stat['success']))
            wandb_log['success'] = stat['success']
        if 'steps_taken' in stat.keys():
            print('Steps-Taken: {:.2f}'.format(stat['steps_taken']))
            wandb_log['steps_taken'] = stat['steps_taken']
        if 'epsilon' in stat.keys():
            print('Epsilon: {:.2f}'.format(stat['epsilon']))
            wandb_log['epsilon'] = stat['epsilon']

        # DroneScatter keys
        if 'pairwise_distance' in stat.keys():
            print('Average pairwise distance: {:.2f}'.format(stat['pairwise_distance']))
            wandb_log['pairwise_distance'] = stat['pairwise_distance']

        # BoxPushing keys
        if 'energy_used' in stat.keys():
            print('Energy used: {:.2f}'.format(stat['energy_used']))
            wandb_log['energy_used'] = stat['energy_used']
        if 'num_exertions' in stat.keys():
            print('Number of exertions: {:.2f}'.format(stat['num_exertions']))
            wandb_log['num_exertions'] = stat['num_exertions']
        if 'num_spills' in stat.keys():
            print('Number of spills: {:.2f}'.format(stat['num_spills']))
            wandb_log['num_spills'] = stat['num_spills']
        if 'ratio_boxes_cleared' in stat.keys():
            print('Ratio of boxes cleared: {:.2f}'.format(stat['ratio_boxes_cleared']))
            wandb_log['ratio_boxes_cleared'] = stat['ratio_boxes_cleared']
        if 'small_boxes_cleared' in stat.keys():
            print('Small boxes cleared: {:.2f}'.format(stat['small_boxes_cleared']))
            wandb_log['small_boxes_cleared'] = stat['small_boxes_cleared']
        if 'large_boxes_cleared' in stat.keys():
            print('Large boxes cleared: {:.2f}'.format(stat['large_boxes_cleared']))
            wandb_log['large_boxes_cleared'] = stat['large_boxes_cleared']

        if 'comm_action' in stat.keys():
            print('Comm-Action: {}'.format(stat['comm_action']))
            wandb_log['comm_action_sum'] = sum(stat['comm_action'])
        if 'enemy_comm' in stat.keys():
            print('Enemy-Comm: {}'.format(stat['enemy_comm']))
        if 'density1' in stat.keys():
            print('density1: {:.4f}'.format(stat['density1']))
        if 'density2' in stat.keys():
            print('density2: {:.4f}'.format(stat['density2']))

        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                             win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

        if args.save_every and ep and args.save and ep % args.save_every == 0:
            save(final=False, episode=ep)

        if args.save:
            save(final=True)

        # Possibly log to wandb
        if args.use_wandb:
            wandb.log(wandb_log)
            # if ep == num_epochs - 1 or (ep + 1) % args.wandb_log_interval == 0:  # Log on last step or interval
            #     wandb.log(wandb_log, step=ep, commit=True)
            # else:
            #     wandb.log(wandb_log, step=ep, commit=False)


def save(final,
         episode=0
         ):
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    if final:
        torch.save(d, run_dir / 'model.pt')
    else:
        torch.save(d, run_dir / ('model_ep%i.pt' % (episode)))


def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])


def signal_handler(signal,
                   frame
                   ):
    print('You pressed Ctrl+C! Exiting gracefully.')
    if args.display:
        env.end_display()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if args.load != '':
    load(args.load)

run(args.num_epochs)
if args.display:
    env.end_display()

if args.save:
    save(final=True)

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()

    os._exit(0)
