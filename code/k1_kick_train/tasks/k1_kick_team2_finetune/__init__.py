import gymnasium as gym

from .env_cfg import K1KickEnvCfg, K1KickEnvCfg_PLAY
from .ppo_cfg import K1KickPPORunnerCfg

gym.register(
    id="Booster-K1-Kick-vp_finetune",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1KickEnvCfg,
        "rsl_rl_cfg_entry_point": K1KickPPORunnerCfg,
    },
)

gym.register(
    id="Booster-K1-Kick-vp-Play_finetune",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1KickEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": K1KickPPORunnerCfg,
    },
)