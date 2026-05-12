"""Gymnasium registration — train (Booster-ID) plus Isaac-style aliases."""

from __future__ import annotations

import gymnasium as gym

_TRAIN_KWARGS = {
    "env_cfg_entry_point": (
        "booster_train.tasks.k1_kick_isaaclab.tasks.locomotion.k1_kick.k1_kick_env_cfg:K1KickEnvCfg"
    ),
    "rsl_rl_cfg_entry_point": (
        "booster_train.tasks.k1_kick_isaaclab.tasks.locomotion.k1_kick.k1_kick_agent_cfg:K1KickPPORunnerCfg"
    ),
}

_PLAY_KWARGS = {
    "env_cfg_entry_point": (
        "booster_train.tasks.k1_kick_isaaclab.tasks.locomotion.k1_kick.k1_kick_env_cfg:K1KickEnvCfg_PLAY"
    ),
    "rsl_rl_cfg_entry_point": (
        "booster_train.tasks.k1_kick_isaaclab.tasks.locomotion.k1_kick.k1_kick_agent_cfg:K1KickPPORunnerCfg"
    ),
}

_STAGE1_V2_TRAIN_KWARGS = {
    "env_cfg_entry_point": (
        "booster_train.tasks.k1_kick_isaaclab.tasks.locomotion.k1_kick.k1_kick_env_cfg:K1KickStableApproachV2EnvCfg"
    ),
    "rsl_rl_cfg_entry_point": (
        "booster_train.tasks.k1_kick_isaaclab.tasks.locomotion.k1_kick.k1_kick_agent_cfg:K1KickStableApproachV2PPORunnerCfg"
    ),
}

_STAGE1_V2_PLAY_KWARGS = {
    "env_cfg_entry_point": (
        "booster_train.tasks.k1_kick_isaaclab.tasks.locomotion.k1_kick.k1_kick_env_cfg:K1KickStableApproachV2EnvCfg_PLAY"
    ),
    "rsl_rl_cfg_entry_point": (
        "booster_train.tasks.k1_kick_isaaclab.tasks.locomotion.k1_kick.k1_kick_agent_cfg:K1KickStableApproachV2PPORunnerCfg"
    ),
}


gym.register(
    id="Booster-K1-Kick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs=_TRAIN_KWARGS.copy(),
)


gym.register(
    id="k1_kick_isaaclab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs=_TRAIN_KWARGS.copy(),
)


gym.register(
    id="k1_kick_isaaclab_play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs=_PLAY_KWARGS.copy(),
)

gym.register(
    id="Booster-K1-Kick-Stable-Approach-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs=_STAGE1_V2_TRAIN_KWARGS.copy(),
)

gym.register(
    id="k1_kick_stable_approach_v2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs=_STAGE1_V2_TRAIN_KWARGS.copy(),
)

gym.register(
    id="k1_kick_stable_approach_v2_play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs=_STAGE1_V2_PLAY_KWARGS.copy(),
)
