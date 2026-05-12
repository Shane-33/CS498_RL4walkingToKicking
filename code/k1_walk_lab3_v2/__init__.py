import gymnasium as gym

gym.register(
    id="Booster-K1-Walk-Lab3-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "booster_train.tasks.k1_walk_lab3_v2.env_cfg:K1WalkEnvCfgV2",
        "rsl_rl_cfg_entry_point": "booster_train.tasks.k1_walk_lab3_v2.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Walk-Lab3-v2-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "booster_train.tasks.k1_walk_lab3_v2.env_cfg:K1WalkEnvCfgV2_PLAY",
        "rsl_rl_cfg_entry_point": "booster_train.tasks.k1_walk_lab3_v2.ppo_cfg:PPORunnerCfg",
    },
)
