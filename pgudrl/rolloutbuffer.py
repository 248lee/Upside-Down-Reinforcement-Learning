import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, rollout, gamma, gae_lambda, opt_lambda):
        states = rollout["states"]
        actions = rollout["actions"]
        log_probs = rollout["log_probs"]
        rewards = rollout["rewards"]
        values = rollout["state_values"]
        next_values = rollout["next_values"]
        desired_returns = rollout["desired_returns"]
        dones = rollout["dones"]
        optimistic_values = rollout["optimistic_values"]
        next_optimistic_values = rollout["next_optimistic_values"]
        
        assert len(states) == len(actions) == len(log_probs) == len(rewards) == len(values) == len(next_values) == len(desired_returns) == len(dones) == len(optimistic_values) == len(next_optimistic_values), "All lists must have the same length"
        # We do not accept torch as input, only NumPy arrays
        # Check if all inputs are NumPy arrays
        for arr in [states, actions, log_probs, rewards, values, next_values, desired_returns]:
            if not isinstance(arr[0], np.ndarray):
                raise TypeError(f"Expected NumPy array, got {type(arr[0])}")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.opt_lambda = opt_lambda

        # Convert to NumPy arrays
        self.states = np.array(states)
        self.actions = np.array(actions)
        self.log_probs = np.array(log_probs)
        self.rewards = np.array(rewards)
        self.values = np.array(values)
        self.next_values = np.array(next_values)
        self.desired_returns = np.array(desired_returns)
        self.dones = np.array(dones)
        self.optimistic_values = np.array(optimistic_values)
        self.next_optimistic_values = np.array(next_optimistic_values)

        # Compute advantages (GAE) BEFORE permutation
        self.advantages = np.zeros_like(self.rewards, dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(len(self.rewards))):
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * self.next_values[t] * next_non_terminal - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam

        # Compute optimistic value targets
        self.opt_targets = np.zeros_like(self.rewards, dtype=np.float32)
        last_opt_lam = 0
        for t in reversed(range(len(self.rewards))):
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * self.next_optimistic_values[t] * next_non_terminal - self.optimistic_values[t]
            last_opt_lam = delta + self.gamma * self.opt_lambda * next_non_terminal * last_opt_lam
            self.opt_targets[t] = last_opt_lam + self.optimistic_values[t]

        # Initialize random indices for sampling
        self.indices = np.random.permutation(len(self.rewards))

        # Permute all arrays including advantages
        self.states = self.states[self.indices]
        self.actions = self.actions[self.indices]
        self.log_probs = self.log_probs[self.indices]
        # self.rewards = self.rewards[self.indices]
        self.values = self.values[self.indices]
        # self.next_values = self.next_values[self.indices]
        self.desired_returns = self.desired_returns[self.indices]
        self.dones = self.dones[self.indices]
        self.advantages = self.advantages[self.indices]
        self.opt_targets = self.opt_targets[self.indices]

        self.start_index = 0
        print("length of rollout buffer:", len(self.rewards))

    def SampleBatch(self, batch_size, device):
        if batch_size > len(self.indices):
            raise ValueError("Batch size cannot be larger than the number of samples in the buffer.")
        
        # Generate batch indices
        batch_indices = np.arange(self.start_index, self.start_index + batch_size) % len(self.indices)
        self.start_index = (self.start_index + batch_size) % len(self.indices)

        if np.any(np.abs(self.desired_returns[batch_indices]) > 1e9):
            max_val = np.max(np.abs(self.desired_returns[batch_indices]))
            print(f"⚠️  Warning: desired return contains values as large as {max_val:.2e} (>|1e2|).")
        
        # Return sampled data
        return {
            'states': torch.tensor(self.states[batch_indices].astype(np.float32)).to(device),
            'actions': torch.tensor(self.actions[batch_indices].astype(np.float32)).to(device),
            'log_probs': torch.tensor(self.log_probs[batch_indices].astype(np.float32)).to(device),
            # 'rewards': torch.tensor(self.rewards[batch_indices].astype(np.float32)).to(device),
            'values': torch.tensor(self.values[batch_indices].astype(np.float32)).to(device),
            # 'next_values': torch.tensor(self.next_values[batch_indices].astype(np.float32)).to(device),
            'desired_returns': torch.tensor(self.desired_returns[batch_indices].astype(np.float32)).to(device),
            'advantages': torch.tensor(self.advantages[batch_indices].astype(np.float32)).to(device),
            'opt_targets': torch.tensor(self.opt_targets[batch_indices].astype(np.float32)).to(device)
        }
    
    def SampleBatchGivenIndices(self, batch_indices, device):
        return {
            'states': torch.tensor(self.states[batch_indices].astype(np.float32)).to(device),
            'actions': torch.tensor(self.actions[batch_indices].astype(np.float32)).to(device),
            'log_probs': torch.tensor(self.log_probs[batch_indices].astype(np.float32)).to(device),
            # 'rewards': torch.tensor(self.rewards[batch_indices].astype(np.float32)).to(device),
            'values': torch.tensor(self.values[batch_indices].astype(np.float32)).to(device),
            # 'next_values': torch.tensor(self.next_values[batch_indices].astype(np.float32)).to(device),
            'desired_returns': torch.tensor(self.desired_returns[batch_indices].astype(np.float32)).to(device),
            'advantages': torch.tensor(self.advantages[batch_indices].astype(np.float32)).to(device)
        }