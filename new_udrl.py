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

from pgudrl.udmodel import BF
from pgudrl.replaybuffer import ReplayBuffer
import pgudrl.suploss as suploss
import pgudrl.udppo as udppo

from pgudrl.rolloutbuffer import RolloutBuffer

env = gym.make("CartPole-v0")
action_space = env.action_space.n
state_space = env.observation_space.shape[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gamma = 0.9
max_reward = 200
return_scale = 0.02
replay_size = 700
n_warm_up_episodes = 50
n_updates_per_iter = 200
n_episodes_per_iter = 20
top_X_eps = 50
batch_size = 40
learning_rate=1e-4
gae_lambda = 0.9
clip_range = 0.15


# %%
## OBSERVE THE WEIGHTS before training
#for p in bf.parameters():
#    print(p)
    

# %% [markdown]
# ## Training Loop

# %%
# Algorithm 2 - Generates an Episode unsing the Behavior Function:
def generate_episode(bf: BF, desired_return: np.ndarray):    
    """
    Generates more samples for the replay buffer.
    Returns states, actions, log_probs, state_values, next_values, desired_returns, rewards, return_to_go[:-1], dones
    """
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    states = []
    desired_returns = []
    actions = []
    log_probs = []
    rewards = []
    state_values = []
    dones = []
    desired_return_no_grad = desired_return
    
    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        # get action from the behavior function, if any
        if bf is None:  # if warmup, use random actions
            action_no_grad = env.action_space.sample()
            # make action_no_grad, which is a numpy.int64, into an 1d array
            action_no_grad = np.array([action_no_grad], dtype=np.int64).squeeze(0)  # make it 0d
            log_prob_no_grad = None
            state_value_no_grad = None
            desired_return_no_grad = None
        else:
            desired_return = torch.tensor(desired_return_no_grad, dtype=torch.float32).to(device)
            action, log_prob, state_value = bf.action(state_tensor, desired_return)
            action_no_grad = action.detach().cpu().numpy()
            log_prob_no_grad = log_prob.detach().cpu().numpy()
            state_value_no_grad = state_value.detach().cpu().numpy()
        next_state, reward, done, trunc, info = env.step(action_no_grad)
        reward = np.array(reward, dtype=np.float32)  # Ensure reward is a float
        # if reward is a scalar, change it to 1d(vector). This helps me generalize to MORL
        if reward.ndim == 0:
            reward = np.array([reward], dtype=np.float32)
        states.append(state)
        desired_returns.append(desired_return_no_grad)  # This desired return has NO gradient
        log_probs.append(log_prob_no_grad)
        actions.append(action_no_grad)
        rewards.append(reward)
        state_values.append(state_value_no_grad)
        dones.append(done)

        if bf is not None:  # if not warmup
            # Update the desired return
            desired_return_no_grad = (desired_return_no_grad - reward) / gamma

        if done or trunc:
            break 

        state = np.array(next_state)
    
    # Calculate the return-to-go in an reverse manner
    return_to_go = [0] * (len(rewards) + 1)  # +1 to handle the next of the last state, which is conviniently 0
    for t in reversed(range(len(rewards))):
        with torch.no_grad():
            return_to_go[t] = rewards[t] + gamma * return_to_go[t + 1]

    next_values = state_values[1:].copy()
    if trunc:
        _, _, next_value = bf.action(torch.tensor(next_state, dtype=torch.float32).to(device), desired_return.to(device))
        next_value = next_value.detach().cpu().numpy()
    else:
        next_value = np.zeros_like(state_values[-1])  # If not trunc, the next value is 0
    
    next_values.append(next_value)
    return states, actions, log_probs, state_values, next_values, desired_returns, rewards, return_to_go[:-1], dones  # Exclude the last element which is 0

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
        states_rollout = []
        actions_rollout = []
        log_probs_rollout = []
        state_values_rollout = []
        next_values_rollout = []
        desired_returns_rollout = []
        rewards_rollout = []
        dones_rollout = []

        for i in range(n_episodes_per_iter):
            
            # Sample exploratory commands based on buffer
            new_desired_reward = replaybuffer.sampling_exploration(top_X_eps)
            
            (
                states, 
                actions, 
                log_probs, 
                state_values,
                next_values,
                desired_returns, 
                rewards, 
                return_to_goes,
                dones
            ) = generate_episode(bf, desired_return=new_desired_reward)

            ewma_G = 0.05 * return_to_goes[0] + (1 - 0.05) * ewma_G
            replaybuffer.add_sample(states, actions, rewards, return_to_goes)
            
            states_rollout.extend(states)
            actions_rollout.extend(actions)
            log_probs_rollout.extend(log_probs)
            state_values_rollout.extend(state_values)
            next_values_rollout.extend(next_values)
            desired_returns_rollout.extend(desired_returns)
            rewards_rollout.extend(rewards)
            dones_rollout.extend(dones)

        rolloutbuffer = RolloutBuffer(
                states_rollout, 
                actions_rollout, 
                log_probs_rollout, 
                rewards_rollout, 
                state_values_rollout, 
                next_values_rollout, 
                desired_returns_rollout, 
                dones_rollout, 
                gamma=gamma, 
                gae_lambda=gae_lambda
            )
            
        # Calculate losses
        pg_loss_buffer = []
        vl_loss_buffer = []
        bf_loss_buffer = []
        for _ in range(n_updates_per_iter):
            optimizer.zero_grad()

            # Calculate the supervised loss
            # Sample a batch from the replay buffer
            input_states, input_commands, output_array = replaybuffer.create_training_examples(batch_size, device=device)
            bf_loss = suploss.CalculateLoss(bf, input_states, input_commands, output_array)
            bf_loss = torch.tensor(0).to(device)

            # Calculate the policy gradient loss and value loss
            # Sample a batch from the rollout buffer
            batch_of_data = rolloutbuffer.SampleBatch(batch_size, device)
            pg_loss, vl_loss = udppo.CalculateLoss(bf, batch_of_data, clip_range)

            # Combine the losses
            total_loss = bf_loss + pg_loss + vl_loss
            total_loss.backward()
            optimizer.step()

            # Log the loss
            bf_loss_buffer.append(bf_loss.item())
            pg_loss_buffer.append(pg_loss.item())
            vl_loss_buffer.append(vl_loss.item())
        
        pg_loss = np.mean(pg_loss_buffer)
        pg_losses.append(pg_loss)
        vl_loss = np.mean(vl_loss_buffer)
        vl_losses.append(vl_loss)
        bf_loss = np.mean(bf_loss_buffer)
        bf_losses.append(bf_loss)

        wandb.log({
            "pg_loss": pg_loss,
            "vl_loss": vl_loss,
            "bf_loss": bf_loss,
            "pg_update extent": pg_loss_buffer[0] - pg_loss_buffer[-1],
            "vl_update extent": vl_loss_buffer[0] - vl_loss_buffer[-1]
        })
        
        # monitoring desired reward and desired horizon
        new_desired_reward = replaybuffer.sampling_exploration(top_X_eps)
        desired_rewards_history.append(new_desired_reward.item())
        # and test it
        # ep_rewards = evaluate(new_desired_reward)
        all_rewards.append(np.sum(rewards))
        average_100_reward.append(np.mean(all_rewards[-100:]))
        

        print("\rEpisode: {} | Desired Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f} | ewma_G: {:.2f}".format(ep, new_desired_reward.item(), np.mean(all_rewards[-100:]), bf_loss, ewma_G[0]), end="", flush=True)
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
                    "top_X_eps": top_X_eps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
               },
               save_code=True)
    
    # Let's create the behavior and buffer
    replaybuffer = ReplayBuffer(replay_size)
    bf = BF(state_space, action_space, hidden_size=64, return_scale=return_scale, seed=1, device=device).to(device)
    optimizer = optim.Adam(params=bf.parameters(), lr=1e-3)

    # Start training
    # Warm up the replay buffer
    for i in range(n_warm_up_episodes):
        states, actions, _, _, _, _, rewards, return_to_goes, _ = generate_episode(None, None)
        replaybuffer.add_sample(states, actions, rewards, return_to_goes)

    rewards, average, d, ud_loss, pg_loss, vl_loss = run_upside_down(max_episodes=500)

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

    
    env.close()
