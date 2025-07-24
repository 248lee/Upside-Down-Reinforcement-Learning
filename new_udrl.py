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

env = gym.make("CartPole-v0")
action_space = env.action_space.n
state_space = env.observation_space.shape[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gamma = 0.9
max_reward = 200
return_scale = 0.02
replay_size = 700
n_warm_up_episodes = 50
n_updates_per_iter = 500
n_rollout_steps_per_iter = 2000
top_X_eps = 50
batch_size = 40
learning_rate=1e-3
ovn_update_rate = 5e-3
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
    episode = {
        "states": [],
        "desired_returns": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "state_values": [],
        "optimistic_values": [],
        "dones": [],
        "return_to_goes": None,
        "next_values": None,
        "next_optimistic_values": None
    }
    
    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        if bf is None:  # warmup
            action_no_grad = env.action_space.sample()
            action_no_grad = np.array([action_no_grad], dtype=np.int64).squeeze(0)
            log_prob_no_grad = None
            state_value_no_grad = None
            optimistic_value_no_grad = None
            desired_return_no_grad = None
        else:
            with torch.no_grad():
                optimistic_value = ovn(state_tensor)
                desired_return = optimistic_value.detach().clone()
                desired_return_no_grad = desired_return.detach().cpu().numpy()
            action, log_prob, state_value = bf.action(state_tensor, desired_return)
            action_no_grad = action.detach().cpu().numpy()
            log_prob_no_grad = log_prob.detach().cpu().numpy()
            state_value_no_grad = state_value.detach().cpu().numpy()
            optimistic_value_no_grad = optimistic_value.detach().cpu().numpy()
        next_state, reward, done, trunc, info = env.step(action_no_grad)
        reward = np.array(reward, dtype=np.float32)
        if reward.ndim == 0:
            reward = np.array([reward], dtype=np.float32)
        episode["states"].append(state)
        episode["desired_returns"].append(desired_return_no_grad)
        episode["log_probs"].append(log_prob_no_grad)
        episode["actions"].append(action_no_grad)
        episode["rewards"].append(reward)
        episode["state_values"].append(state_value_no_grad)
        episode["optimistic_values"].append(optimistic_value_no_grad)
        episode["dones"].append(done)

        if bf is not None:
            desired_return_no_grad = (desired_return_no_grad - reward) / gamma

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
    episode["next_values"] = episode["state_values"][1:].copy()
    episode["next_optimistic_values"] = episode["optimistic_values"][1:].copy()
    if bf is not None:
        if not done:
            _, _, next_value = bf.action(torch.tensor(next_state, dtype=torch.float32).to(device), desired_return.to(device))
            next_value = next_value.detach().cpu().numpy()
            next_optimistic_value = ovn(torch.tensor(next_state, dtype=torch.float32).to(device))
            next_optimistic_value = next_optimistic_value.detach().cpu().numpy()
        else:
            next_value = np.zeros_like(episode["state_values"][-1])
            next_optimistic_value = np.zeros_like(episode["optimistic_values"][-1])
        episode["next_values"].append(next_value)
        episode["next_optimistic_values"].append(next_optimistic_value)

        episode["T"] = len(episode["rewards"])

    return episode

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
    ewma_T = 0

    for ep in range(1, max_episodes+1):
        rollout = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "state_values": [],
            "next_values": [],
            "desired_returns": [],
            "rewards": [],
            "dones": [],
            "optimistic_values": [],
            "next_optimistic_values": []
        }

        rollout_step_count = 0
        while rollout_step_count < n_rollout_steps_per_iter:
            
            # Sample exploratory commands based on buffer            
            episode = generate_episode(bf, ovn)

            rollout_step_count += episode["T"]
            ewma_G = 0.05 * episode["return_to_goes"][0] + (1 - 0.05) * ewma_G
            ewma_T = 0.05 * episode["T"] + (1 - 0.05) * ewma_T
            replaybuffer.add_sample(episode)
            
            rollout["states"].extend(episode["states"])
            rollout["actions"].extend(episode["actions"])
            rollout["log_probs"].extend(episode["log_probs"])
            rollout["state_values"].extend(episode["state_values"])
            rollout["next_values"].extend(episode["next_values"])
            rollout["desired_returns"].extend(episode["desired_returns"])
            rollout["rewards"].extend(episode["rewards"])
            rollout["dones"].extend(episode["dones"])
            rollout["optimistic_values"].extend(episode["optimistic_values"])
            rollout["next_optimistic_values"].extend(episode["next_optimistic_values"])

        rolloutbuffer = RolloutBuffer(
                rollout,
                gamma=gamma, 
                gae_lambda=gae_lambda,
                opt_lambda=opt_lambda
            )
            
        # Calculate losses
        pg_loss_buffer = []
        vl_loss_buffer = []
        bf_loss_buffer = []
        for iter in range(n_updates_per_iter):
            optimizer_bf.zero_grad()
            optimizer_ovn.zero_grad()

            # Calculate the supervised loss
            # Sample a batch from the replay buffer
            input_states, input_commands, output_array = replaybuffer.create_training_examples(batch_size, device=device)
            bf_loss = suploss.CalculateLoss(bf, input_states, input_commands, output_array)
            # bf_loss = torch.tensor(0).to(device)

            # Calculate the policy gradient loss and value loss
            # Sample a batch from the rollout buffer
            batch_of_data = rolloutbuffer.SampleBatch(batch_size, device)
            pg_loss, vl_loss = udppo.CalculateLoss(bf, batch_of_data, clip_range)

            # Calculate the loss of optimistic value network
            ovn_loss = ovntrain.CalculateLoss(ovn, batch_of_data)

            if iter == 0:
                with torch.no_grad():
                    testing_batch_of_data = rolloutbuffer.SampleBatchGivenIndices(np.linspace(0, 300, 15, dtype=np.int64), device)
                    pg_loss_start, vl_loss_start = udppo.CalculateLoss(bf, testing_batch_of_data, clip_range)
            elif iter == n_updates_per_iter - 1:
                with torch.no_grad():
                    testing_batch_of_data = rolloutbuffer.SampleBatchGivenIndices(np.linspace(0, 300, 15, dtype=np.int64), device)
                    pg_loss_end, vl_loss_end = udppo.CalculateLoss(bf, testing_batch_of_data, clip_range)
            else:
                pass

            # Combine the losses
            total_loss = bf_loss + pg_loss + vl_loss
            total_loss.backward()
            optimizer_bf.step()

            # Update ovn
            ovn_loss.backward()
            optimizer_ovn.step()

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
            "pg_update extent": pg_loss_start - pg_loss_end,
            "vl_update extent": vl_loss_start - vl_loss_end
        })
        
        # monitoring desired reward and desired horizon
        desired_rewards_history.append(episode["desired_returns"][0])
        # and test it
        # ep_rewards = evaluate(new_desired_reward)
        all_rewards.append(np.sum(episode["rewards"]))
        average_100_reward.append(np.mean(all_rewards[-100:]))
        

        print("\rEpisode: {} | Desired Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f} | ewma_T: {} | ewma_G: {:.2f}".format(ep, episode["desired_returns"][0][0], np.mean(all_rewards[-100:]), bf_loss, int(ewma_T), ewma_G[0]), end="", flush=True)
        if ep % 100 == 0:
            print("\rEpisode: {} | Desired Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f}".format(ep, episode["desired_returns"][0][0], np.mean(all_rewards[-100:]), bf_loss))
            
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
                    "n_rollout_steps_per_iter": n_rollout_steps_per_iter,
                    "top_X_eps": top_X_eps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
               },
               save_code=True)
    
    # Let's create the behavior and buffer
    replaybuffer = ReplayBuffer(replay_size)
    bf = BF(state_space, action_space, hidden_size=64, return_scale=return_scale, seed=1, device=device).to(device)
    optimizer_bf = optim.Adam(params=bf.parameters(), lr=learning_rate)

    ovn = OptimisticValueNetwork(state_space, action_space, hidden_size=64, seed=1, device=device).to(device)
    optimizer_ovn = optim.SGD(params=ovn.parameters(), lr=ovn_update_rate)

    # Start training
    # Warm up the replay buffer
    for i in range(n_warm_up_episodes):
        episode = generate_episode(None, None)
        replaybuffer.add_sample(episode)

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
