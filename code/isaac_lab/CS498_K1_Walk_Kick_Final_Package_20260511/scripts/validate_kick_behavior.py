import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv

from booster_train.tasks.k1_kick_isaaclab.tasks.locomotion.k1_kick.mdp.kick_features import (
    get_kick_features,
)

CHECKPOINT = "/home/team2/IsaacLab/logs/rsl_rl/k1_kick_isaaclab/2026-05-09_19-55-49_kick_fixed_ball_v1/model_1999.pt"

NUM_STEPS = 2000

# =========================================================
# env
# =========================================================

env_cfg = parse_env_cfg(
    "Booster-K1-Kick-v0",
    device="cuda:0",
    num_envs=1,
)

env = ManagerBasedRLEnv(cfg=env_cfg)

# =========================================================
# checkpoint
# =========================================================

ckpt = torch.load(CHECKPOINT, map_location="cuda:0")
state = ckpt["model_state_dict"]

# =========================================================
# rebuild actor
# =========================================================

actor = nn.Sequential(
    nn.Linear(86, 256),
    nn.ELU(),
    nn.Linear(256, 128),
    nn.ELU(),
    nn.Linear(128, 128),
    nn.ELU(),
    nn.Linear(128, 20),
).cuda()

actor.load_state_dict({
    k.replace("actor.", ""): v
    for k, v in state.items()
    if k.startswith("actor.")
})

actor.eval()

obs_mean = state["actor_obs_normalizer._mean"].cuda()
obs_std = state["actor_obs_normalizer._std"].cuda()

# =========================================================
# rollout stats
# =========================================================

right_swings = 0
left_swings = 0
contact_events = 0
forward_ball_events = 0
falls = 0

obs, _ = env.reset()

for step in range(NUM_STEPS):

    obs_tensor = obs["policy"]

    obs_norm = (obs_tensor - obs_mean) / (obs_std + 1e-8)

    with torch.no_grad():
        actions = actor(obs_norm)

    obs, _, terminated, truncated, _ = env.step(actions)

    f = get_kick_features(env)

    right_foot_vx = f.right_foot_vel_w[0, 0].item()
    left_foot_vx = f.left_foot_vel_w[0, 0].item()

    ball_vx = f.ball_vel_w[0, 0].item()

    d_right = f.d_right_ball[0].item()

    base_height = env.scene["robot"].data.root_pos_w[0, 2].item()

    if right_foot_vx > 0.5:
        right_swings += 1

    if left_foot_vx > 0.5:
        left_swings += 1

    if d_right < 0.20:
        contact_events += 1

    if ball_vx > 0.3:
        forward_ball_events += 1

    if base_height < 0.45:
        falls += 1

# =========================================================
# results
# =========================================================

print("\n================ VALIDATION RESULTS ================\n")

print(f"right swing events : {right_swings}")
print(f"left swing events  : {left_swings}")

print()

print(f"ball-near events   : {contact_events}")
print(f"ball-forward events: {forward_ball_events}")

print()

print(f"falls              : {falls}")

print("\n====================================================\n")

simulation_app.close()
