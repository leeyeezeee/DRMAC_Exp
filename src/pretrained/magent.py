import os
from types import SimpleNamespace as SN
from collections import OrderedDict

import torch

from modules.agents.rnn_agent import RNNAgent


def _convert_keys(original_dict):
    new_dict = OrderedDict()
    for key, value in original_dict.items():
        if "agents.0." in key:
            continue
        elif "agents.1." in key:
            # only load
            new_key = key.replace("agents.1.", "")
            new_dict[new_key] = value
        else:
            raise ValueError(f"Unexpected key: {key}")
    return new_dict


class IDQN_Battle:

    def __init__(self, pretrained_ckpt, input_shape):
        args = SN(use_rnn=False, hidden_dim=64, n_actions=21)

        # The CUDA runtime does not support the fork start method; either the spawn or forkserver start method are required to use CUDA in subprocesses.
        self.device = "cpu"
        self.agent = RNNAgent(input_shape, args).to(self.device)

        model_path = os.path.join(os.path.dirname(__file__), pretrained_ckpt)
        save_dict = torch.load(model_path, map_location=self.device)
        self.agent.load_state_dict(save_dict)
        self.agent.eval()
        self.hidden_states = self.agent.init_hidden()  # a positional argument, no use

    def step(self, obss, avail_actions):
        with torch.no_grad():
            obss = torch.tensor(obss, device=self.device, dtype=torch.float32)
            avail_actions = torch.tensor(avail_actions, device=self.device)
            agent_outputs, _ = self.agent.forward(obss, self.hidden_states)

            agent_outputs[avail_actions == 0.0] = -float("inf")
            actions = agent_outputs.max(dim=-1)[1]
        return actions


class IDQN_AdvPursuit(IDQN_Battle):
    def __init__(self, pretrained_ckpt, input_shape):
        args = SN(use_rnn=False, hidden_dim=64, n_actions=13)

        # The CUDA runtime does not support the fork start method; either the spawn or forkserver start method are required to use CUDA in subprocesses.
        self.device = "cpu"
        self.agent = RNNAgent(input_shape, args).to(self.device)

        model_path = os.path.join(os.path.dirname(__file__), pretrained_ckpt)
        save_dict = torch.load(model_path, map_location=self.device)
        save_dict = _convert_keys(save_dict)
        self.agent.load_state_dict(save_dict)
        self.agent.eval()
        self.hidden_states = self.agent.init_hidden()  # a positional argument, no use
