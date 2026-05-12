import gymnasium as gym

gym.register(
    id="Booster-K1-FullBody-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.k1_fullbody_walk_env_cfg:K1FullBodyWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.k1_fullbody_walk_agent_cfg:K1FullBodyWalkPPORunnerCfg",
    },
)
