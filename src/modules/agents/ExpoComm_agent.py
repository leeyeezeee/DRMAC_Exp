# code adapted from https://github.com/wendelinboehmer/dcg
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ExpoCommSAgent(nn.Module):
    def __init__(self, input_shape, args):
        nn.Module.__init__(self)
        self.args = args
        self.n_agents = args.n_agents
        self.attention_dim = args.attention_dim

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        assert self.args.use_rnn

        self.msg_k = nn.Linear(args.hidden_dim, args.attention_dim)
        self.msg_q = nn.Linear(args.hidden_dim, args.attention_dim)
        self.msg_v = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.n_actions),
        )

        self.state_dim = int(np.prod(args.state_shape))
        self.predict_net = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, self.state_dim),
        )

    def init_hidden(self):
        # make hidden states on same device as model
        h = self.fc1.weight.new(1, self.args.hidden_dim).zero_()
        msg = th.zeros_like(h)
        hidden_info = (h, msg)
        return hidden_info

    def forward(self, inputs, hidden_info, topk_indices):

        hidden_state, msg = hidden_info
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)

        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, h_in)
        msg = self._communicate(topk_indices, msg, h)

        # detach msg for main loss, use aux loss for msg
        out = th.cat([h, msg], dim=-1)
        q = self.fc2(out)
        # (bs * n_agents, n_actions), (bs * n_agents, hidden_dim)
        hidden_info = (h, msg)
        return q, hidden_info, msg

    def aux_forward(self, msg):
        return self.predict_net(msg)

    def _communicate(self, topk_indices, other_msg, ego_h):
        bs, n_agents, topk = topk_indices.shape
        assert n_agents == self.n_agents
        # (bs, n_agents, topk, hidden_dim)
        topk_indices = topk_indices[:, :, :, None].expand(
            -1, -1, -1, self.args.hidden_dim
        )
        # (bs, n_agents(repeated), n_agents, hidden_dim)
        other_msg = other_msg.reshape(bs, 1, n_agents, self.args.hidden_dim).expand(
            -1, n_agents, -1, -1
        )

        # (bs, n_agents, topk, hidden_dim)
        msg_received = other_msg.gather(dim=2, index=topk_indices)

        ego_h = ego_h.reshape(bs, n_agents, -1)
        q = self.msg_q(ego_h).reshape(bs, n_agents, self.attention_dim, 1)
        k = self.msg_k(msg_received)
        # (bs, n_agents, topk)
        attention = th.matmul(k, q)[:, :, :, 0]
        attention = F.softmax(attention / math.sqrt(self.attention_dim), dim=-1)
        # (bs, n_agents, topk, hidden_dim)
        m = self.msg_v(msg_received)
        # (bs, n_agents, hidden_dim)
        m_aggregated = (attention[:, :, :, None] * m).sum(dim=2)

        return m_aggregated.reshape(bs * n_agents, -1)


class ExpoCommOAgent(ExpoCommSAgent):
    def __init__(self, input_shape, args):
        nn.Module.__init__(self)
        self.args = args
        self.n_agents = args.n_agents
        self.hidden_dim = args.hidden_dim

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        assert self.args.use_rnn

        self.msg_processor = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.ReLU(),
        )
        self.msg_rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)

        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.n_actions),
        )

        self.state_dim = int(np.prod(args.state_shape))

        self.predict_net = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, self.state_dim),
        )

    def _communicate(self, topk_indices, other_msg, ego_h):
        bs, n_agents = topk_indices.shape
        assert n_agents == self.n_agents
        # (bs, n_agents, 1, hidden_dim)
        topk_indices = topk_indices[:, :, None, None].expand(
            -1, -1, -1, self.args.hidden_dim
        )

        msg_ego = other_msg.reshape(bs * n_agents, self.args.hidden_dim)
        # (bs, n_agents(repeated), n_agents, hidden_dim)
        other_msg = other_msg.reshape(bs, 1, n_agents, self.args.hidden_dim).expand(
            -1, n_agents, -1, -1
        )

        # (bs, n_agents, 1, hidden_dim)
        msg_received = other_msg.gather(dim=2, index=topk_indices)
        msg_received = msg_received[:, :, 0, :]

        # (bs, n_agents, hidden_dim)
        ego_h = ego_h.reshape(bs, n_agents, -1)
        msg_input = self.msg_processor(th.cat([ego_h, msg_received], dim=-1))

        msg_input = msg_input.reshape(bs * n_agents, -1)

        m_aggregated = self.msg_rnn(msg_input, msg_ego)

        return m_aggregated.reshape(bs * n_agents, -1)


class ExpoCommOContAgent(ExpoCommOAgent):

    def aux_forward(self, msg):
        return F.normalize(msg, dim=-1)


class ExpoCommSContAgent(ExpoCommSAgent):

    def aux_forward(self, msg):
        return F.normalize(msg, dim=-1)
