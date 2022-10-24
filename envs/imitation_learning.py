import numpy as np

using_imitation = False
num_normal_experiences = 0
num_imitation_experiences = 0


def use_imitation(args):
    if not args.imitation:
        return False

    global using_imitation, num_imitation_experiences, num_normal_experiences

    if using_imitation:
        num_imitation_experiences += 1
        if num_imitation_experiences >= args.num_imitation_experiences:
            using_imitation = False
            num_imitation_experiences = 0
    else:
        num_normal_experiences += 1
        if num_normal_experiences >= args.num_normal_experiences:
            using_imitation = True
            num_normal_experiences = 0

    return using_imitation


def get_experience(wrapped_env, env_name):
    if env_name == "box_pushing":
        return get_box_pushing_experiences(wrapped_env)
    else:
        raise Exception("Unsupported env for imitation learning: ", env_name)


def get_box_pushing_experiences(wrapped_env):
    env = wrapped_env.env
    experiences = []

    # Actions: 0: STAY, 1: UP, 2: RIGHT, 3: DOWN, 4: LEFT, 5: UP_POW, 6: RIGHT_POW, 7: DOWN_POW, 8: LEFT_POW
    possible_actions = tuple(range(1, 9))
    for action in possible_actions:
        # Manually spawn in a box type (0 for no, 1 for large, 2 for small)
        if action < 5:  # Small boxes
            env._manual_boxes = 2
            env.reset()
        else:  # Large box
            env._manual_boxes = 1
            env.reset()

        # Sample number of steps = width of environment from environment
        for step in range(env.dims[0]):
            if env.episode_over:
                break

            if env.small_box_positions is not None:
                actions = [np.full(env.nrobots, action)]
            else:
                actions = [np.full(env.nrobots, action)]
            experiences.append(wrapped_env.step(actions))

    return experiences
