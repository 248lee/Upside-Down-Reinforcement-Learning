import threading
import tkinter as tk
from tkinter import ttk

import torch
import gymnasium as gym
import numpy as np

from pgudrl.udmodel import BF
from pgudrl.ovntrain import OptimisticValueNetwork


# ---------------------------
# UI: Slider for "rate"
# ---------------------------
class RateController:
    def __init__(self, root, init_value=0.4):
        self._lock = threading.Lock()
        self._val = float(init_value)

        frm = ttk.Frame(root, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="UDRL rate (0–1):").pack(anchor="w")

        self.var = tk.DoubleVar(value=self._val)

        def on_change(_=None):
            v = float(self.var.get())
            with self._lock:
                self._val = v
            value_lbl.config(text=f"{v:.2f}")

        slider = ttk.Scale(frm, from_=0.0, to=1.0, orient="horizontal",
                           variable=self.var, command=on_change)
        slider.pack(fill="x", padx=4, pady=8)

        value_lbl = ttk.Label(frm, text=f"{self._val:.2f}", width=6)
        value_lbl.pack(anchor="e")

        # small hint
        ttk.Label(frm, text="Close this window to stop the run.").pack(anchor="w", pady=(8, 0))

    def get(self) -> float:
        with self._lock:
            return self._val


# ---------------------------
# RL runner (background thread)
# ---------------------------
def run_test(stop_event: threading.Event, get_rate):
    # Gym / device
    env = gym.make("LunarLander-v3", render_mode="human")
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load models
    bf = BF(state_space, action_space, hidden_size=256, return_scale=0.05,
            gamma=0.99, seed=1, device=device, replay_buffer=None).to(device)
    bf.load_state_dict(torch.load("bf_model_300.pth", map_location=device))
    bf.eval()

    ovn = OptimisticValueNetwork(state_space, action_space, hidden_size=64,
                                 gamma=0.99, seed=1, device=device).to(device)
    ovn.load_state_dict(torch.load("ovn_model_300.pth", map_location=device))
    ovn.eval()

    try:
        with torch.no_grad():
            for ep in range(50):
                if stop_event.is_set():
                    break

                state, _ = env.reset()
                state = np.array(state, dtype=np.float32)
                first_state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

                # initial desired return from OVN and current rate
                optimistic_value, pesimistic_value = ovn(first_state_tensor)
                rate = (1 - float(get_rate()))
                desired_return = optimistic_value - rate * (optimistic_value - pesimistic_value)

                total_reward = 0.0

                while True:
                    if stop_event.is_set():
                        break

                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

                    # refresh rate and clamp desired return to the optimistic band
                    optimistic_value, pesimistic_value = ovn(state_tensor)
                    rate = (1 - float(get_rate()))
                    band_low = optimistic_value - rate * (optimistic_value - pesimistic_value)
                    # Align desired return within the optimistic band randomly
                    epsilon = np.random.uniform(0.0, 1.0)
                    if epsilon < 0.25:
                        desired_return = torch.clamp(desired_return, min=band_low.item(), max=optimistic_value.item())
                    else:
                        desired_return = torch.clamp(desired_return, max=optimistic_value.item())

                    # policy forward
                    action_logits, q_value = bf(state_tensor, desired_return)
                    action = torch.argmax(action_logits)
                    action_no_grad = int(action.detach().cpu().numpy())

                    next_state, reward, done, trunc, info = env.step(action_no_grad)
                    total_reward += float(reward)

                    # update desired return from chosen action's q and observed reward
                    desired_return = q_value[action.unsqueeze(0)] - reward
                    desired_return = desired_return / bf.gamma
                    desired_return = desired_return.to(device)

                    state = np.array(next_state, dtype=np.float32)

                    # Optional: render (commented)
                    # env.render()

                    if done or trunc:
                        break

                print(f"Episode {ep+1}: total_reward = {total_reward:.2f}")

    finally:
        env.close()


def main():
    # Tk must run in the main thread; RL loop goes to a worker thread
    root = tk.Tk()
    root.title("UDRL Rate Controller")

    # (Optional) nicer default ttk theme if available
    try:
        from tkinter import ttk
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    controller = RateController(root, init_value=1.0)

    stop_event = threading.Event()
    worker = threading.Thread(target=run_test, args=(stop_event, controller.get), daemon=True)
    worker.start()

    def on_close():
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

    # Wait for the RL thread to finish after the window closes
    worker.join(timeout=5)


if __name__ == "__main__":
    main()
