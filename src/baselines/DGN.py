import math, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class Encoder(nn.Module):
    def __init__(self,
                 din=32,
                 hidden_dim=128
                 ):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)

    def forward(self,
                x
                ):
        embedding = F.relu(self.fc(x))
        return embedding


class AttModel(nn.Module):
    def __init__(self,
                 n_node,
                 din,
                 hidden_dim,
                 dout
                 ):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self,
                x,
                mask
                ):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        att = F.softmax(torch.mul(torch.bmm(q, k), mask) - 9e15 * (1 - mask), dim=2)

        out = torch.bmm(att, v)
        # out = torch.add(out,v)
        out = F.relu(self.fcout(out))
        return out


class Q_Net(nn.Module):
    def __init__(self,
                 hidden_dim,
                 dout
                 ):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self,
                x
                ):
        q = self.fc(x)
        return q


class DGN(nn.Module):

    def __init__(self,
                 args,
                 num_inputs
                 ):
        super(DGN, self).__init__()

        # Ability to scale number of comm passes provided as augmentation to original model
        self.comm_passes = args.comm_passes

        self.num_inputs = num_inputs

        n_agent = args.nagents
        hidden_dim = args.hid_size
        self.num_actions = args.naction_heads[0]  # Hack due to multi action nonsense

        self.encoder = Encoder(num_inputs, hidden_dim)

        # Create message-passing models
        self.att_models = []
        assert self.comm_passes >= 1, "Need at least 1 message-passing step for DGN"
        for _ in range(self.comm_passes):
            self.att_models.append(AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim))
        self.att_models = torch.nn.ModuleList(self.att_models)

        self.q_net = Q_Net(hidden_dim, self.num_actions)

    def forward(self,
                x,
                info
                ):
        mask = info["env_graph"]
        mask = torch.Tensor(mask)  # TODO: figure out better place to do this

        h = self.encoder(x)

        for comm_pass in range(self.comm_passes):
            h = self.att_models[comm_pass](h, mask)

        q = self.q_net(h)

        return q
