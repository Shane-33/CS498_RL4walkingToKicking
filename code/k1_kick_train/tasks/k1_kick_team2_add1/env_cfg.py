from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils

from booster_train.assets.robots.booster import BOOSTER_K1_CFG
from .managers_cfg import CommandsCfg, ActionsCfg, ObservationsCfg, RewardsCfg, TerminationsCfg, EventsCfg


@configclass
class K1KickSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    # Contact sensor covers both feet — used for single_leg_balance and
    # kick_foot_forward_swing. The right-foot-only sensor for ball_contact
    # is declared separately in managers_cfg with body_names=["right_foot_link"].
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot_link",
        history_length=6,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )
    # Ball — FIFA standard size 5 (radius 0.11m, mass 0.43kg).
    # Init position is the fallback; EventsCfg randomizes it to
    # x∈[0.25, 0.45], y∈[-0.2, 0.0] each episode reset.
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.8),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, -0.1, 0.11),
        ),
    )


@configclass
class K1KickEnvCfg(ManagerBasedRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=0.002, render_interval=10)
    scene: K1KickSceneCfg = K1KickSceneCfg(num_envs=4096, env_spacing=2.5)
    decimation: int = 10
    episode_length_s: float = 10.0
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(4.0, -4.0, 3.0),
        lookat=(0.0, 0.0, 0.5),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = BOOSTER_K1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=BOOSTER_K1_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.58),
                joint_pos={
                    ".*_Hip_Pitch":      -0.2,
                    ".*_Hip_Roll":        0.0,
                    ".*_Hip_Yaw":         0.0,
                    ".*_Knee_Pitch":      0.4,
                    ".*_Ankle_Pitch":    -0.25,
                    ".*_Ankle_Roll":      0.0,
                    ".*_Shoulder_Pitch":  0.0,
                    ".*_Shoulder_Roll":   0.0,
                    ".*_Elbow_Pitch":     0.0,
                    ".*_Elbow_Yaw":       0.0,
                },
                joint_vel={".*": 0.0},
            ),
        )


@configclass
class K1KickEnvCfg_PLAY(K1KickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5