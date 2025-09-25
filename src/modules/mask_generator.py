import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMask(nn.Module):
    def __init__(self,args, input_shape, threshold=0.3, use_rnn=False):
        super(FeatureMask, self).__init__()
        self.args = args
        self.latent_dim = args.state_repre_dim * args.n_agents
        self.fc1 = nn.Linear(input_shape, self.latent_dim)

    def forward(self, inputs, hidden_state=None):
        x = self.fc1(inputs)
        x = F.sigmoid(x)        
        return(x)
        
