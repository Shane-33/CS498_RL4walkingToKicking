import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from booster_train.assets.robots.booster import BOOSTER_K1_CFG, K1_ACTION_SCALE

from .managers_cfg import ActionsCfg, ObservationsCfg, CommandsCfg, EventCfg, RewardsCfg, TerminationsCfg


@configclass
class K1SceneCfg(InteractiveSceneCfg):
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.57),
            joint_pos={
                "Left_Shoulder_Roll":  -1.3,
                "Right_Shoulder_Roll":  1.3,
                "Left_Elbow_Pitch":     0.3,
                "Right_Elbow_Pitch":    0.3,
                "AAHead_yaw":           0.0,
                "Head_pitch":          -0.03,
            },
            joint_vel={".*": 0.0},
        ),
    )


@configclass
class K1WalkEnvCfgV2(ManagerBasedRLEnvCfg):
    scene: K1SceneCfg = K1SceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    viewer = ViewerCfg(eye=(8.0, 8.0, 3.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.actions.joint_pos.scale = dict(K1_ACTION_SCALE)


@configclass
class K1WalkEnvCfgV2_PLAY(K1WalkEnvCfgV2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 10.0),
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            heading_command=True,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.6, 0.6),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=(-3.14159, 3.14159),
            ),
        )
