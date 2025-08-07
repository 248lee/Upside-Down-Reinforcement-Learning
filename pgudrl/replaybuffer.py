import numpy as np
import torch
from typing import NamedTuple
from gymnasium import spaces
import pickle
from sklearn.preprocessing import StandardScaler

class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    return_to_goes: torch.Tensor

class ReplayBuffer():

    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str
    ):
        self.buffer = {
            "observations": np.zeros((buffer_size, *observation_space.shape), dtype=np.float32),
            "next_observations": np.zeros((buffer_size, *observation_space.shape), dtype=np.float32),
            "actions": np.zeros((buffer_size, *action_space.shape), dtype=np.float32),
            "rewards": np.zeros((buffer_size, 1), dtype=np.float32),
            "dones": np.zeros((buffer_size, 1), dtype=np.float32),
            "return_to_goes": np.zeros((buffer_size, 1), dtype=np.float32),
        }

        self.pos = 0  # This is the pointer of the next sample to be added
        self.buffer_size = buffer_size  # This is the maximum size of the buffer
        self.full = False  # This is a flag to indicate if the buffer is full
        self.device = device

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)
        
        
    def add_sample(self, episode):
        # Check that `episode` is a dictionary with the expected keys
        for key in self.buffer.keys():
            if key not in episode:
                raise ValueError(f"Episode must contain the key: {key}")
                return
            episode[key] = np.array(episode[key])  # Convert the list to nd-array

        length_of_episode = len(episode["observations"])
        # add the episode to the buffer
        if self.pos + length_of_episode >= self.buffer_size:  # If the episode is longer than the remaining space in the buffer
            for key in self.buffer.keys():
                # Firstly, fill up whole buffer
                self.buffer[key][self.pos:self.buffer_size] = episode[key][:self.buffer_size - self.pos]
                # Next, fill the rest of the episode into the beginning of the buffer
                self.buffer[key][:length_of_episode - (self.buffer_size - self.pos)] = episode[key][self.buffer_size - self.pos:]
            self.full = True  # If the buffer is full, set the flag to True
            self.pos = length_of_episode - (self.buffer_size - self.pos)  # Reset the position to the end of the episode
        else:
            for key in self.buffer.keys():
                self.buffer[key][self.pos:self.pos + length_of_episode] = episode[key]
            self.pos += length_of_episode
        
    
    def SampleBatch(self, batch_size):
        # self.sort()
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        self.Standardlize()
        data = (
            (self.buffer["observations"][batch_inds]),
            self.buffer["actions"][batch_inds],
            self.buffer["next_observations"][batch_inds],
            self.buffer["dones"][batch_inds],
            self.buffer["rewards"][batch_inds],
            (self.buffer["return_to_goes"][batch_inds]),
        )
        return ReplayBufferSamples(*map(self.to_torch, data))

    def Standardlize(self):
        if self.is_calculated_mean_and_std == False:
            scaler = StandardScaler()
            scaler.fit(self.buffer["observations"])
            self.buffer["observations"] = scaler.transform(self.buffer["observations"])
            self.buffer["next_observations"] = scaler.transform(self.buffer["next_observations"])
            scalar = StandardScaler()
            scalar.fit(self.buffer["return_to_goes"])
            self.buffer["return_to_goes"] = scalar.transform(self.buffer["return_to_goes"])
            self.is_calculated_mean_and_std = True
    
    def GetAllData(self):
        data = (
            (self.buffer["observations"]),
            self.buffer["actions"],
            (self.buffer["next_observations"]),
            self.buffer["dones"],
            self.buffer["rewards"],
           ( self.buffer["return_to_goes"]),
        )
        return ReplayBufferSamples(*map(self.to_torch, data))
    
    
    def __len__(self):
        return len(self.buffer_size) if self.full else self.pos
    
    def sampling_exploration(self, top_X_eps):
        """
        This function calculates the new desired reward based on the replay buffer.
        New desired return-to-go is sampled from a uniform distribution given the mean and the std calculated from the last best X performances.
        where X is the hyperparameter last_few.
        
        """
        
        top_X = self.get_nbest(top_X_eps)
        # save all top_X cumulative returns in a list 
        returns = [i["return_to_goes"][0] for i in top_X]
        # from these returns calc the mean and std
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)
        # sample desired reward from a uniform distribution given the mean and the std
        new_desired_reward = np.random.uniform(mean_returns, mean_returns+std_returns, size=1)

        return new_desired_reward
    
    def create_training_examples(self, batch_size, device):
        """
        Creates a data set of training examples that can be used to create a data loader for training.
        ============================================================
        1. for the given batch_size episode idx are randomly selected
        2. based on these episodes t1 and t2 are samples for each selected episode 
        3. for the selected episode and sampled t1 and t2 trainings values are gathered
        ______________________________________________________________
        Output are two numpy arrays in the length of batch size:
        Input Array for the Behavior function - consisting of (state, desired_return_to_go, time_horizon)
        Output Array with the taken actions 
        """
        input_states = []
        input_commands = []
        output_array = []
        # select randomly episodes from the buffer
        episodes = self.get_random_samples(batch_size)
        for ep in episodes:
            #select time stamps
            t = select_time_steps(ep)
            state, desired_return_to_go, action = create_training_input(ep, t)
            input_states.append(torch.tensor(state, dtype=torch.float32).to(device))
            input_commands.append(torch.tensor(desired_return_to_go, dtype=torch.float32).to(device))
            output_array.append(torch.tensor(action, dtype=torch.float32).to(device))
        return input_states, input_commands, output_array

    def save_replay_buffer(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    

def select_time_steps(saved_episode):
    """
    Given a saved episode from the replay buffer this function samples a random time step in that episode:
    T = max time horizon in that episode
    Returns t
    """
    # Select times in the episode:
    T = len(saved_episode["observations"]) # episode max horizon 
    t = np.random.randint(0,T-1)

    return t

def create_training_input(episode, t):
    """
    Based on the selected episode and the given time steps this function returns 4-1 values:
    1. state at t
    2. the desired reward: sum over all rewards from t to the end of the episode
    3. the time horizon T (legacy)
    4. the target action taken at t
    
    buffer episodes are build like [cumulative episode reward, observations, actions, rewards]
    """
    state = episode["observations"][t] 
    desired_return_to_go = episode["return_to_goes"][t]  # this is the cumulative return from t to the end of the episode
    action = episode["actions"][t]
    return state, desired_return_to_go, action
