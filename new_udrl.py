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
from pgudrl.ovntrain import OptimisticValueNetwork
from pgudrl.replaybuffer import ReplayBuffer
import pgudrl.suploss as suploss
import pgudrl.udppo as udppo
import pgudrl.ovntrain as ovntrain

from pgudrl.rolloutbuffer import RolloutBuffer

env = gym.make("LunarLander-v3")
action_space = env.action_space.n
state_space = env.observation_space.shape[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gamma = 0.98
max_reward = 200
return_scale = 1
replay_size = 100000
n_warm_up_episodes = 50
ovn_n_updates_per_iter = 200
n_rollout_steps_per_iter = 520
top_X_eps = 50
batch_size = 512
learning_rate=3e-4
ovn_update_rate = 1e-5
gae_lambda = 0.9
opt_lambda = 0.9
clip_range = 0.2


# %%
## OBSERVE THE WEIGHTS before training
#for p in bf.parameters():
#    print(p)
    

# %% [markdown]
# ## Training Loop

# %%
# Algorithm 2 - Generates an Episode unsing the Behavior Function:
def generate_episode(bf: BF, ovn: OptimisticValueNetwork):
    """
    Generates more samples for the replay buffer.
    Returns a dictionary containing episode data.
    """
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    initial_state_tensor = torch.tensor(state.copy(), dtype=torch.float32).to(device)
    episode = {
        "observations": [],
        "next_observations": [],
        # "desired_returns": [],
        "actions": [],
        # "log_probs": [],
        "rewards": [],
        # "state_values": [],
        # "optimistic_values": [],
        "dones": [],
        "return_to_goes": None,
        # "next_values": None,
        # "next_optimistic_values": None
    }
    
    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        # epsilon greedy exploration
        epsilon = 0.01

        if bf is None or np.random.rand() < epsilon:  # warmup
            action_no_grad = env.action_space.sample()
            action_no_grad = np.array([action_no_grad], dtype=np.int64).squeeze(0)
            # log_prob_no_grad = None
            # state_value_no_grad = None
            # optimistic_value_no_grad = None
            # desired_return_no_grad = None
        else:
            with torch.no_grad():
                optimistic_value = ovn(state_tensor)
                desired_return = optimistic_value.detach().clone()
                # desired_return_no_grad = desired_return.detach().cpu().numpy()
            action, log_prob, state_value = bf.action(state_tensor, desired_return)
            action_no_grad = action.detach().cpu().numpy()
            # log_prob_no_grad = log_prob.detach().cpu().numpy()
            # state_value_no_grad = state_value.detach().cpu().numpy()
            # optimistic_value_no_grad = optimistic_value.detach().cpu().numpy()
        next_state, reward, done, trunc, info = env.step(action_no_grad)
        reward = np.array(reward, dtype=np.float32)
        if reward.ndim == 0:
            reward = np.array([reward], dtype=np.float32)
        episode["observations"].append(state)
        episode["next_observations"].append(next_state)
        # episode["desired_returns"].append(desired_return_no_grad)
        # episode["log_probs"].append(log_prob_no_grad)
        episode["actions"].append(action_no_grad)  # This list adds a dimension to the actions
        episode["rewards"].append(reward)
        # episode["state_values"].append(state_value_no_grad)
        # episode["optimistic_values"].append(optimistic_value_no_grad)
        episode["dones"].append([1.0] if done else [0.0])

        # if bf is not None:
        #     desired_return_no_grad = (desired_return_no_grad - reward) / gamma

        if done or trunc:
            break

        state = np.array(next_state)

    # Calculate return-to-go
    rewards = episode["rewards"]
    return_to_goes = [0] * (len(rewards) + 1)
    for t in reversed(range(len(rewards))):
        with torch.no_grad():
            return_to_goes[t] = rewards[t] + gamma * return_to_goes[t + 1]
    episode["return_to_goes"] = return_to_goes[:-1]

    # Calculate next_values and next_optimistic_values
    # episode["next_values"] = episode["state_values"][1:].copy()
    # episode["next_optimistic_values"] = episode["optimistic_values"][1:].copy()
    # if bf is not None:
    #     if not done:
    #         _, _, next_value = bf.action(torch.tensor(next_state, dtype=torch.float32).to(device), desired_return.to(device))
    #         next_value = next_value.detach().cpu().numpy()
    #         next_optimistic_value = ovn(torch.tensor(next_state, dtype=torch.float32).to(device))
    #         next_optimistic_value = next_optimistic_value.detach().cpu().numpy()
    #     else:
    #         # next_value = np.zeros_like(episode["state_values"][-1])
    #         next_optimistic_value = np.zeros_like(episode["optimistic_values"][-1])
        # episode["next_values"].append(next_value)
        # episode["next_optimistic_values"].append(next_optimistic_value)

    episode["T"] = len(episode["rewards"])
    if ovn is not None:
        with torch.no_grad():
            episode["first_desired_return"] = ovn(initial_state_tensor).detach().cpu().numpy()

    return episode

# Algorithm 1 - Upside - Down Reinforcement Learning 
def run_upside_down(max_episodes):
    """
    
    """
    all_rewards = []
    bf_losses = []
    explore_losses = []
    q_losses = []
    average_100_reward = []
    desired_rewards_history = []
    ewma_G = 0
    ewma_T = 0
    store_replay_buffer = False

    for ep in range(1, max_episodes+1):
        # rollout = {
        #     "observations": [],
        #     "rewards": [],
        #     "dones": [],
        #     "next_observations": []
        # }

        rollout_step_count = 0
        while rollout_step_count < n_rollout_steps_per_iter:
            
            # Sample exploratory commands based on buffer            
            episode = generate_episode(bf, ovn)

            rollout_step_count += episode["T"]
            ewma_G = 0.05 * episode["return_to_goes"][0] + (1 - 0.05) * ewma_G
            ewma_T = 0.05 * episode["T"] + (1 - 0.05) * ewma_T
            replaybuffer.add_sample(episode)
            
        #     rollout["observations"].extend(episode["observations"])
        #     rollout["rewards"].extend(episode["rewards"])
        #     rollout["dones"].extend(episode["dones"])
        #     rollout["next_observations"].extend(episode["next_observations"])

        # rolloutbuffer = RolloutBuffer(rollout, gamma=gamma)
            
        # Calculate losses
        explore_loss_buffer = []
        q_loss_buffer = []
        bf_loss_buffer = []
        accuracy_buffer = []

        if store_replay_buffer or ep == 800:
            replaybuffer.save_replay_buffer("observe_sup_loss")
            torch.save(bf.state_dict(), "bf_model.pth")
            torch.save(ovn.state_dict(), "ovn_model.pth")

        for iter in range(max(int(ewma_T), n_rollout_steps_per_iter)):
            optimizer_bf.zero_grad()
            optimizer_ovn.zero_grad()

            # Calculate the supervised loss
            # Sample a batch from the replay buffer
            batch = replaybuffer.SampleBatch(batch_size)
            bf_loss, explore_loss, q_loss, ovn_loss = suploss.CalculateLoss(bf, ovn, batch)
            accuracy, _, _ = suploss.test_accuracy(bf, batch)
            # bf_loss = torch.tensor(0).to(device)

            # Combine the losses
            total_loss = bf_loss + explore_loss + q_loss
            total_loss.backward()
            optimizer_bf.step()

            # Update the optimistic value network
            ovn_loss.backward()
            optimizer_ovn.step()

            # Log the loss
            bf_loss_buffer.append(bf_loss.item())
            explore_loss_buffer.append(explore_loss.item())
            q_loss_buffer.append(q_loss.item())
            accuracy_buffer.append(accuracy)

            
        explore_loss = np.mean(explore_loss_buffer)
        explore_losses.append(explore_loss)
        q_loss = np.mean(q_loss_buffer)
        q_losses.append(q_loss)
        bf_loss = np.mean(bf_loss_buffer)
        bf_losses.append(bf_loss)
        accuracy = np.mean(accuracy_buffer)

        
        
        # monitoring desired reward and desired horizon
        desired_rewards_history.append(episode["first_desired_return"])
        # and test it
        # ep_rewards = evaluate(new_desired_reward)
        all_rewards.append(np.sum(episode["rewards"]))
        average_100_reward.append(np.mean(all_rewards[-100:]))

        wandb.log({
            "explore_loss": explore_loss,
            "q_loss": q_loss,
            "bf_loss": bf_loss,
            "ewma_G": ewma_G,
            "accuracy": accuracy,
            "desired_reward": episode["first_desired_return"][0],
            "mean_100_rewards": np.mean(all_rewards[-100:]),
        })
        

        print("\rEpisode: {} | Desired Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f} | ewma_T: {} | ewma_G: {:.2f}".format(ep, episode["first_desired_return"][0], np.mean(all_rewards[-100:]), bf_loss, int(ewma_T), ewma_G[0]), end="", flush=True)
        if ep % 100 == 0:
            print("\rEpisode: {} | Desired Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f}".format(ep, episode["first_desired_return"][0], np.mean(all_rewards[-100:]), bf_loss))
            
    return all_rewards, average_100_reward, desired_rewards_history, bf_losses, explore_losses, q_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Let's write in wandb!
    wandb.init(project="PGUDRL",
               config={
                    "max_reward": max_reward,
                    "return_scale": return_scale,
                    "replay_size": replay_size,
                    "n_warm_up_episodes": n_warm_up_episodes,
                    "n_rollout_steps_per_iter": n_rollout_steps_per_iter,
                    "top_X_eps": top_X_eps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
               },
               save_code=True)
    
    # Let's create the behavior and buffer
    replaybuffer = ReplayBuffer(replay_size, env.observation_space, env.action_space, device)
    bf = BF(state_space, action_space, hidden_size=256, return_scale=return_scale, gamma=gamma, replay_buffer=replaybuffer,seed=1, device=device).to(device)
    optimizer_bf = optim.Adam(params=bf.parameters(), lr=learning_rate)

    ovn = OptimisticValueNetwork(state_space, action_space, hidden_size=64, gamma=gamma, seed=1, device=device).to(device)
    optimizer_ovn = optim.SGD(params=ovn.parameters(), lr=ovn_update_rate)

    # Start training
    # Warm up the replay buffer
    for i in range(n_warm_up_episodes):
        episode = generate_episode(None, None)
        replaybuffer.add_sample(episode)

    rewards, average, d, ud_loss, explore_losses, q_losses = run_upside_down(max_episodes=50000)

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
    plt.title("Explore Loss")
    plt.plot(explore_losses)
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
