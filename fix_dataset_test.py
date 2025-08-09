import wandb
import pickle
import gymnasium as gym
import torch.optim as optim
import torch
from pgudrl.udmodel import BF
from pgudrl.ovntrain import OptimisticValueNetwork
from pgudrl.replaybuffer import ReplayBuffer
import pgudrl.suploss as suploss
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

learning_rate=3e-4
ovn_update_rate = 1e-5
batch_size = 512

return_scale = 1
gamma = 0.98
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("LunarLander-v3")
action_space = env.action_space.n
state_space = env.observation_space.shape[0]

wandb.init(project="PGUDRL_fixed_dataset",
            config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            save_code=True)

# Let's create the behavior and buffer
with open("observe_sup_loss", 'rb') as f:
    replaybuffer = pickle.load(f)
    replaybuffer.is_calculated_mean_and_std = False

bf = BF(state_space, action_space, hidden_size=256, return_scale=return_scale, gamma=gamma, replay_buffer=replaybuffer, seed=1, device=device).to(device)
is_load_bf = input("BF: are you going to load the bf_model? (y/)")
if is_load_bf == "y":
    bf.load_state_dict(torch.load("bf_model.pth"))
else:
    pass

optimizer_bf = optim.Adam(params=bf.parameters(), lr=learning_rate)

ovn = OptimisticValueNetwork(state_space, action_space, hidden_size=64, gamma=gamma, seed=1, device=device).to(device)
is_load_ovn = input("OVN: are you going to load the ovn_model? (y/)")
if is_load_ovn == "y":
    ovn.load_state_dict(torch.load("ovn_model.pth"))
else:
    pass
optimizer_ovn = optim.SGD(params=ovn.parameters(), lr=ovn_update_rate)

for iter in tqdm(range(500 * 400)):
    optimizer_bf.zero_grad()
    optimizer_ovn.zero_grad()

    # Calculate the supervised loss
    # Sample a batch from the replay buffer
    batch = replaybuffer.SampleBatch(batch_size)
    bf_loss, explore_loss, q_loss, ovn_loss = suploss.CalculateLoss(bf, ovn, batch)
    accuracy, _, _ = suploss.test_accuracy(bf, batch)
    # bf_loss = torch.tensor(0).to(device)

    # Combine the losses
    total_loss = bf_loss #+ explore_loss + q_loss
    total_loss.backward()
    optimizer_bf.step()

    # Update the optimistic value network
    ovn_loss.backward()
    optimizer_ovn.step()

    # Log the loss
    wandb.log({
        "explore_loss": explore_loss.item(),
        "q_loss": q_loss.item(),
        "bf_loss": bf_loss.item(),
        "accuracy": accuracy,
    })

data = replaybuffer.GetAllData()
accuracy, y, yhat = suploss.test_accuracy(bf, data)

print("accuracy:", accuracy)
y = y.cpu().numpy()
yhat = yhat.cpu().numpy()
cm = confusion_matrix(yhat, y)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Action 0', 'Action 1', 'Action 2', 'Action 3'])
disp.plot() # Use a colormap like Blues
plt.title("Confusion Matrix")
plt.savefig('Confusion Matrix.png')