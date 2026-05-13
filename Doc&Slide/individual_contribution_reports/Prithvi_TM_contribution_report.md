# Individual Contribution Report

**Name:** Rithvi Prem  
**Team:** Team 2  
**Course:** CS 498 Robotics Team Project, Spring 2026  
**Project:** Reinforcement Learning for Humanoid Walking-to-Kicking

## Individual Contributions

I contributed to the design, implementation, training, debugging, and evaluation of our humanoid walking-to-kicking reinforcement learning project throughout the semester.

My main contributions included:

1. **Reinforcement Learning Pipeline Setup and Troubleshooting**
   - Set up and maintained the Isaac Lab training pipeline, including environment registration, checkpoint management, and training launch infrastructure.
   - Diagnosed and resolved a wide range of technical issues including conflicts across multiple repositories.

2. **Walking Policy Development and Tuning**
   - Supported the design and iterative refinement of the humanoid walking policy reward structure.
   - Investigated sources of policy stiffness and jitter, and proposed curriculum-based reward shaping strategies to encourage more natural locomotion.
   - Analyzed differences between reference implementations and our policy, identifying key design decisions around gait clocks, reward scaling, and domain randomization.

3. **Kicking Policy Development**
   - Led the design and implementation of the kicking task from scratch, including scene setup, ball physics, reward engineering, and staged training curricula.
   - Developed and iterated on reward functions covering ball approach, contact quality, shot direction, post-kick recovery, COM stability, single-leg balance, and full kick cycle completion.
   - Implemented domain randomization for sim-to-real transfer including observation noise, pose randomization, push disturbances, and friction variation.
   - Designed termination conditions and reward structures to prevent degenerate behaviors such as kamikaze kicks, ground skimming, and lateral collapse.

4. **Experiment Tracking and Evaluation**
   - Monitored training progress using TensorBoard forwarded to local machines, comparing reward curves across multiple training runs and checkpoints.
   - Generated and evaluated rollout videos across training stages to qualitatively assess policy improvement.
   - Systematically compared policy behaviors across checkpoints to guide reward redesign decisions.

## Summary

My work spanned the full technical lifecycle of the project — from infrastructure setup and debugging through reward design, training, and evaluation. I focused particularly on making the kicking policy training pipeline robust and interpretable, and on bridging the gap between walking stability and purposeful ball interaction.