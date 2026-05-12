K1 Final Tasks and Checkpoints Package

Included task directories:
1. tasks/k1_fullbody_walk_isaaclab
   - Full-body K1 walking task in Isaac Lab.

2. tasks/k1_kick_isaaclab
   - K1 soccer kicking task in Isaac Lab with fixed-ball kicking setup.

Primary walking checkpoint:
- checkpoints/walk/model_46999.pt
- Source run:
  /home/team2/team2_project/booster_train/scripts/rsl_rl/logs/rsl_rl/k1_walk/2026-04-28_16-28-43_ppo

Primary kicking checkpoint:
- checkpoints/kick/k1_kick_isaaclab_2026-05-09_21-22-44_kick_fixed_ball_v1/model_1999.pt
- Source run:
  /home/team2/team2_project/booster_train/logs/rsl_rl/k1_kick_isaaclab/2026-05-09_21-22-44_kick_fixed_ball_v1

Optional archived staged kicking checkpoints are included under:
- checkpoints/kick/k1_kick_team2_stage3_from2999
- checkpoints/kick/k1_kick_team2_singlestage_fromstage2

Evaluation / utility scripts are under:
- scripts/

This package intentionally excludes unrelated Hydra output folders, Python cache files, and unused task directories.
