import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
import math
import torch.nn.functional as F

# ---------- building blocks ----------
class FiLM(nn.Module):
    """Produce per-feature (gamma, beta) from the command vector."""
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, 2*hidden_dim)
        )
    def forward(self, c):
        g, b = self.net(c).chunk(2, dim=-1)
        return g, b

class CondResBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1   = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.drop  = nn.Dropout(drop)
        # one FiLM generator per pre-act
        self.film1 = FiLM(cond_dim, hidden_dim)
        self.film2 = FiLM(cond_dim, hidden_dim)

    def forward(self, x, c):
        # pre-act + FiLM 1
        h = self.norm1(x)
        g1, b1 = self.film1(c)
        h = (1 + g1) * h + b1
        h = F.silu(self.fc1(h))
        h = self.drop(h)

        # pre-act + FiLM 2
        h = self.norm2(h)
        g2, b2 = self.film2(c)
        h = (1 + g2) * h + b2
        h = self.fc2(h)
        return x + h  # residual

# ---------- full model ----------
class BF(nn.Module):
    """
    Supervised classifier (actor logits) optionally with a critic head.
    - State:  R^state_space
    - Command: R^cmd_dim (e.g., desired return); FiLM-conditioning applied in every block.
    """
    def __init__(self, state_space, action_space, return_scale, gamma, replay_buffer, hidden_size=256, n_blocks=4,
                 cmd_dim=1, dropout=0.1, with_critic=True, seed=0, device="cpu"):
        super().__init__()
        torch.manual_seed(seed)
        self.device = device
        self.with_critic = with_critic
        self.return_scale = return_scale
        self.gamma = gamma
        self.replay_buffer = replay_buffer

        # towers
        self.state_in = nn.Sequential(
            nn.Linear(state_space, hidden_size),
            nn.SiLU(),
        )
        self.cmd_in = nn.Sequential(
            nn.Linear(cmd_dim, max(32, hidden_size // 4)),
            nn.SiLU(),
        )
        cond_dim = max(32, hidden_size // 4)

        # conditioned residual trunk
        blocks = [CondResBlock(hidden_size, cond_dim, drop=dropout) for _ in range(n_blocks)]
        self.trunk = nn.ModuleList(blocks)

        # heads
        self.actor_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_space)
        )
        if with_critic:
            self.critic_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.SiLU(),
                nn.Linear(hidden_size // 2, action_space)
            )

        # Kaiming init for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state, command):
        x = self.state_in(state)
        c = self.cmd_in(command)
        for block in self.trunk:
            x = block(x, c)
        actor_logits = self.actor_head(x)
        if self.with_critic:
            critic_scores = self.critic_head(x)
            return actor_logits, critic_scores
        else:
            return actor_logits
    
    def action(self, state, desire):
        """
        Samples the action based on their probability
        """
        command = (desire*self.return_scale)
        action_prob, q_value = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action), q_value
    
    def forward_to_get_logprob_and_q_value(self, state, desire, action):
        """
        Returns the log probability of the action and the state value
        """
        command = (desire*self.return_scale)
        action_prob, q_value = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        m = Categorical(probs)
        log_prob = m.log_prob(action)
        return log_prob, q_value[action]
    
    def greedy_action(self, state, desire):
        """
        Returns the greedy action 
        """
        command = (desire*self.return_scale).unsqueeze(0)  # Ensure command is a 2D tensor
        action_prob, _ = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        action = torch.argmax(probs).item()
        return action