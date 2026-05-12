from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class K1KickPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50

    experiment_name = "k1_kick_team2"
    run_name = "ppo_stage1"

    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,          # conservative — preserve standing behavior
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,          # lower than walk — fine-tuning not scratch
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )