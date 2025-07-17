# %%
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import gymnasium as gym
import copy
import wandb
import matplotlib.pyplot as plt
import argparse

# %%
# init Environment
# env = gym.make("LunarLander-v3")
env = gym.make("CartPole-v0")
action_space = env.action_space.n
state_space = env.observation_space.shape[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gamma = 0.9
max_reward = 200
return_scale = 0.02
replay_size = 700
n_warm_up_episodes = 50
n_updates_per_iter = 100
n_episodes_per_iter = 15
last_few = 50
batch_size = 256
learning_rate=1e-3

# %% [markdown]
# ## Behavior Function

# %%
class BF(nn.Module):
    def __init__(self, state_space, action_space, hidden_size, seed):
        super(BF, self).__init__()
        torch.manual_seed(seed)
        self.actions = np.arange(action_space)
        self.action_space = action_space
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.commands = nn.Linear(1, hidden_size)  # Here, we assume 1 command: desired return
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.actor_fc4 = nn.Linear(hidden_size, hidden_size)
        self.actor_fc5 = nn.Linear(hidden_size, action_space)
        
        self.critic_fc4 = nn.Linear(hidden_size, hidden_size)
        self.critic_fc5 = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, state, command):       
               
        out = self.sigmoid(self.fc1(state))
        command_out = self.sigmoid(self.commands(command))
        out = out * command_out
        out = torch.relu(self.fc2(out))
        feature = torch.relu(self.fc3(out))

        # Actor
        actor_out = torch.relu(self.actor_fc4(feature))
        actor_out = self.actor_fc5(actor_out)

        # Critic
        critic_out = torch.relu(self.critic_fc4(feature))
        critic_out = self.critic_fc5(critic_out)
        
        return actor_out, critic_out
    
    def action(self, state, desire):
        """
        Samples the action based on their probability
        """
        command = (desire*return_scale).unsqueeze(0)  # Ensure command is a 2D tensor
        action_prob, state_value = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action), state_value
    
    def greedy_action(self, state, desire):
        """
        Returns the greedy action 
        """
        command = (desire*return_scale).unsqueeze(0)  # Ensure command is a 2D tensor
        action_prob, _ = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        action = torch.argmax(probs).item()
        return action

# %% [markdown]
# ## Replay Buffer

# %%
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

# %%
## OBSERVE THE WEIGHTS before training
#for p in bf.parameters():
#    print(p)

# %%
# FUNCTIONS FOR Sampling exploration commands

def sampling_exploration( top_X_eps = last_few):
    """
    This function calculates the new desired reward based on the replay buffer.
    New desired return-to-go is sampled from a uniform distribution given the mean and the std calculated from the last best X performances.
    where X is the hyperparameter last_few.
    
    """
    
    top_X = buffer.get_nbest(last_few)
    # save all top_X cumulative returns in a list 
    returns = [i["return_to_goes"][0] for i in top_X]
    # from these returns calc the mean and std
    mean_returns = np.mean(returns)
    std_returns = np.std(returns)
    # sample desired reward from a uniform distribution given the mean and the std
    new_desired_reward = np.random.uniform(mean_returns, mean_returns+std_returns)

    return torch.tensor([new_desired_reward], dtype=torch.float32)

# %%
# FUNCTIONS FOR TRAINING
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

def create_training_examples(batch_size):
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
    input_array = []
    output_array = []
    # select randomly episodes from the buffer
    episodes = buffer.get_random_samples(batch_size)
    for ep in episodes:
        #select time stamps
        t = select_time_steps(ep)
        state, desired_return_to_go, action = create_training_input(ep, t)
        input_array.append(torch.cat([state, torch.tensor([desired_return_to_go], dtype=torch.float32)]))
        output_array.append(action)
    return input_array, output_array

def train_behavior_function(batch_size):
    """
    Trains the BF with on a cross entropy loss were the inputs are the action probabilities based on the state and command.
    The targets are the actions appropriate to the states from the replay buffer.
    """
    optimizer.zero_grad()
    X, y = create_training_examples(batch_size)

    X = torch.stack(X)
    state = X[:,0:state_space]
    d = X[:,state_space:state_space+1]
    command = d*return_scale
    y = torch.stack(y).long().squeeze(-1)  # Convert y to be a 1D tensor
    y_ = bf(state.to(device), command.to(device))[0].float()  # Get only the action probabilities (the first output of the BF)
    pred_loss = F.cross_entropy(y_, y)
    pred_loss.backward()
    optimizer.step()
    return pred_loss.detach().cpu().numpy()

# %%
def evaluate(desired_return: torch.Tensor):
    """
    Runs one episode of the environment to evaluate the bf.
    """
    state, _ = env.reset()
    rewards = 0
    while True:
        state = torch.tensor(state, dtype=torch.float32)
        action, _, _ = bf.action(state.to(device), desired_return.to(device))
        state, reward, done, trunc, info = env.step(action.item()) 
        rewards += reward
        desired_return = min(desired_return - reward, torch.tensor([max_reward], dtype=torch.float32))
        
        if done or trunc:
            break 
    return rewards
    

# %% [markdown]
# ## Training Loop

# %%
# Algorithm 2 - Generates an Episode unsing the Behavior Function:
def generate_episode(bf: BF, desired_return: torch.Tensor):    
    """
    Generates more samples for the replay buffer.
    Returns states, actions, log_probs, state_values, desired_returns, rewards, return_to_go[:-1]
    """
    state, _ = env.reset()
    states = []
    desired_returns = []
    actions = []
    log_probs = []
    rewards = []
    state_values = []
    
    while True:
        state = torch.tensor(state, dtype=torch.float32)
        # get action from the behavior function, if any
        if bf is None:  # if warmup, use random actions
            action = env.action_space.sample()
            next_state, reward, done, trunc, info = env.step(action)
            action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)  # unsqueeze to make it 2D
            log_prob = None
            state_value = None
            desired_return_no_grad = None
        else:
            action, log_prob, state_value = bf.action(state.to(device), desired_return.to(device))
            desired_return_no_grad = desired_return.item()
            next_state, reward, done, trunc, info = env.step(action.item())
        states.append(state)
        desired_returns.append(desired_return_no_grad)  # This desired return has NO gradient
        log_probs.append(log_prob)
        actions.append(action)
        rewards.append(reward)
        state_values.append(state_value)

        state = next_state

        if bf is not None:  # if not warmup
            # Update the desired return
            desired_return = (desired_return - reward) / gamma
        
        if done or trunc:
            break 
    
    # Calculate the return-to-go in an reverse manner
    return_to_go = [0] * (len(rewards) + 1)  # +1 to handle the next of the last state, which is conviniently 0
    for t in reversed(range(len(rewards))):
        with torch.no_grad():
            return_to_go[t] = rewards[t] + gamma * return_to_go[t + 1]
    return states, actions, log_probs, state_values, desired_returns, rewards, return_to_go[:-1]  # Exclude the last element which is 0

def calculate_loss_online_PG(states, rewards, log_probs, state_values, desired_returns, return_to_goes):
    assert (len(states) == len(log_probs) and len(log_probs) == len(desired_returns) and len(desired_returns) == len(return_to_goes)), "Lengths of all lists should be equal"
    optimizer.zero_grad()
    total_actor_loss = 0
    total_critic_loss = 0

    # Calculate the losses
    for t in range(len(states)):
        with torch.no_grad():
            next_value = state_values[t + 1].item() if t < len(state_values) - 1 else 0
            target = (rewards[t] + gamma * next_value)
        # Let's compare whether the desired_returns is equal to the states
        total_actor_loss += (state_values[t].item() - desired_returns[t]) * return_to_goes[t] * log_probs[t]
        total_critic_loss += ((state_values[t] - target)**2)
    total_critic_loss /= len(states)
    loss = total_actor_loss + total_critic_loss
        
    # Update parameters
    loss.backward()
    optimizer.step()

    return total_actor_loss.detach().cpu().numpy(), total_critic_loss.detach().cpu().numpy()


# Algorithm 1 - Upside - Down Reinforcement Learning 
def run_upside_down(max_episodes):
    """
    
    """
    all_rewards = []
    bf_losses = []
    pg_losses = []
    vl_losses = []
    average_100_reward = []
    desired_rewards_history = []
    ewma_G = 0
    for ep in range(1, max_episodes+1):

        # improve|optimize bf based on replay buffer
        bf_loss_buffer = []
        for i in range(n_updates_per_iter):
            bf_loss = train_behavior_function(batch_size)
            bf_loss_buffer.append(bf_loss)
            wandb.log({"bf_loss": bf_loss})
        bf_loss = np.mean(bf_loss_buffer)
        # bf_loss = 0
        bf_losses.append(bf_loss)
        
        # run x new episode and add to buffer
        pg_loss_buffer = []
        vl_loss_buffer = []

        for i in range(n_episodes_per_iter):
            
            # Sample exploratory commands based on buffer
            new_desired_reward = sampling_exploration()
            (
                states, 
                actions, 
                log_probs, 
                state_values, 
                desired_returns, 
                rewards, 
                return_to_goes
            ) = generate_episode(bf, desired_return=new_desired_reward)

            # pg_loss, vl_loss = calculate_loss_online_PG(states, rewards, log_probs, state_values, desired_returns, return_to_goes)
            # pg_loss_buffer.append(pg_loss)
            # vl_loss_buffer.append(vl_loss)
            ewma_G = 0.05 * return_to_goes[0] + (1 - 0.05) * ewma_G
            buffer.add_sample(states, actions, rewards, return_to_goes)
            
        pg_loss = np.mean(pg_loss_buffer)
        pg_losses.append(pg_loss)
        vl_loss = np.mean(vl_loss_buffer)
        vl_losses.append(vl_loss)
        new_desired_reward = sampling_exploration()
        # monitoring desired reward and desired horizon
        desired_rewards_history.append(new_desired_reward.item())
        
        ep_rewards = evaluate(new_desired_reward)
        all_rewards.append(ep_rewards)
        average_100_reward.append(np.mean(all_rewards[-100:]))
        wandb.log({
            "desired return": new_desired_reward,
            "actual return": ep_rewards,
            # "bf_loss": bf_loss,
            "pg_loss": pg_loss,
            "vl_loss": vl_loss,
        })

        print("\rEpisode: {} | Desired Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f} | ewma_G: {:.2f}".format(ep, new_desired_reward.item(), np.mean(all_rewards[-100:]), bf_loss, ewma_G), end="", flush=True)
        if ep % 100 == 0:
            print("\rEpisode: {} | Desired Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f}".format(ep, new_desired_reward.item(), np.mean(all_rewards[-100:]), bf_loss))
            
    return all_rewards, average_100_reward, desired_rewards_history, bf_losses, pg_losses, vl_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Let's write in wandb!
    wandb.init(project="PGUDRL",
               config={
                    "max_reward": max_reward,
                    "return_scale": return_scale,
                    "replay_size": replay_size,
                    "n_warm_up_episodes": n_warm_up_episodes,
                    "n_updates_per_iter": n_updates_per_iter,
                    "n_episodes_per_iter": n_episodes_per_iter,
                    "last_few": last_few,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
               },
               save_code=True)
    
    # Let's create the behavior and buffer
    buffer = ReplayBuffer(replay_size)
    bf = BF(state_space, action_space, 64, 1).to(device)
    optimizer = optim.Adam(params=bf.parameters(), lr=1e-3)

    # Start training
    # Warm up the replay buffer
    for i in range(n_warm_up_episodes):
        states, actions, _, _, _, rewards, return_to_goes = generate_episode(None, None)
        buffer.add_sample(states, actions, rewards, return_to_goes)

    rewards, average, d, ud_loss, pg_loss, vl_loss = run_upside_down(max_episodes=50)

    # Finish training
    torch.save(bf.state_dict(), "behaviorfunction.pth")
    plt.figure(figsize=(15,8))
    plt.subplot(2,2,1)
    plt.title("Rewards")
    plt.plot(rewards, label="rewards")
    plt.plot(average, label="average100")
    plt.legend()
    plt.subplot(2,2,2)
    plt.title("UD Loss")
    plt.plot(ud_loss)
    plt.subplot(2,2,3)
    plt.title("desired Rewards")
    plt.plot(d)
    plt.subplot(2,2,4)
    plt.title("PG Loss")
    plt.plot(pg_loss)
    plt.show()

    # %%
    # SAVE MODEL
    name = "model.pth"
    torch.save(bf.state_dict(), name)

    # %%
    ## OBSERVE THE WEIGHTS after training
    #for p in bf.parameters():
    #    print(p)

    # %% [markdown]
    # ## EVALUATION RUN

    # %%
    DESIRED_RETURN_TO_GO = torch.tensor([200], dtype=torch.float32).to(device)
    desired = DESIRED_RETURN_TO_GO.item()

    env.reset()
    total_reward = 0
    while True:
        command = DESIRED_RETURN_TO_GO*return_scale

        probs_logits = bf(torch.from_numpy(state).float().to(device), command)
        probs = torch.softmax(probs_logits, dim=-1).detach().cpu()
        action = torch.argmax(probs).item()
        state, reward, done, trunc, info = env.step(action)
        total_reward += reward
        DESIRED_RETURN_TO_GO -= reward
        if done or trunc:
            break

    print("Desired rewards: {} | after finishing one episode the agent received {} rewards".format(desired, rewards))
    env.close()
