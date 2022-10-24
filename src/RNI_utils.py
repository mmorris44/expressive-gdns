import torch


# Adds either rni or one-hot agent ID
# Add agent ID if rni_num is -1
def augment_state(state, rni_num):
    if rni_num == -1:
        return add_agent_ids_to_state(state)

    batch_dim = state.shape[0]
    agent_dim = state.shape[1]

    bound = 1  # TODO: parameterize
    rni = torch.rand(size=(batch_dim, agent_dim, rni_num))  # Use uniform for now (rand() gives [0, 1])
    # torch.normal can be used for a normal distribution

    # Make RNI in [-bound, bound]
    rni = torch.mul(rni, 2 * bound)  # [0, 2 * bound]
    sub = torch.ones(size=(batch_dim, agent_dim, rni_num))  # {1}
    sub = torch.mul(sub, bound)  # {bound}
    rni = torch.sub(rni, sub)  # [-bound, bound]

    # Create the state
    state = torch.cat((state, rni), dim=2)

    return state


# Add one-hot agent IDs to states
def add_agent_ids_to_state(state):
    batch_dim = state.shape[0]
    agent_dim = state.shape[1]

    agent_ids = torch.zeros(size=(agent_dim, agent_dim))
    for agent_id in range(agent_dim):
        agent_ids[agent_id][agent_id] = 1

    state = torch.cat((state, agent_ids.expand(batch_dim, -1, -1)), dim=2)

    return state
