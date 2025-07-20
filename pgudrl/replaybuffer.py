import numpy as np
import torch

class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        
        
    def add_sample(self, states, actions, rewards, return_to_goes):
        episode = {"states": states, "actions":actions, "rewards": rewards, "return_to_goes": return_to_goes}
        self.buffer.append(episode)
        
    
    def sort(self):
        #sort buffer with the highest return-to-go
        self.buffer = sorted(self.buffer, key = lambda i: i["return_to_goes"][0],reverse=True)
        # keep the max buffer size
        self.buffer = self.buffer[:self.max_size]
    
    def get_random_samples(self, batch_size):
        self.sort()
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        return batch
    
    def get_nbest(self, n):
        self.sort()
        return self.buffer[:n]
    
    def __len__(self):
        return len(self.buffer)
    
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
    

def select_time_steps(saved_episode):
    """
    Given a saved episode from the replay buffer this function samples a random time step in that episode:
    T = max time horizon in that episode
    Returns t
    """
    # Select times in the episode:
    T = len(saved_episode["states"]) # episode max horizon 
    t = np.random.randint(0,T-1)

    return t

def create_training_input(episode, t):
    """
    Based on the selected episode and the given time steps this function returns 4-1 values:
    1. state at t
    2. the desired reward: sum over all rewards from t to the end of the episode
    3. the time horizon T (legacy)
    4. the target action taken at t
    
    buffer episodes are build like [cumulative episode reward, states, actions, rewards]
    """
    state = episode["states"][t] 
    desired_return_to_go = episode["return_to_goes"][t]  # this is the cumulative return from t to the end of the episode
    action = episode["actions"][t]
    return state, desired_return_to_go, action