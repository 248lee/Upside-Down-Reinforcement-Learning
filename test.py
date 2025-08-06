import stable_baselines3.dqn as dqn
import pickle

with open("observe_sup_loss", 'rb') as f:
    buffer = pickle.load(f)

print("finish loading")