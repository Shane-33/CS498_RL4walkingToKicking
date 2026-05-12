# Individual Contribution Report

**Name:** Shanshan Zhu  
**Team:** Team 2  
**Course:** CS 498 Robotics Team Project, Spring 2026  
**Project:** Reinforcement Learning for Humanoid Walking-to-Kicking

## Individual Contributions

I contributed to the design, implementation, training, debugging, evaluation, and documentation of our humanoid walking-to-kicking reinforcement learning project.

My main contributions included:

1. **Isaac Lab task development and debugging**
   - Worked on setting up and debugging Booster K1 humanoid tasks in Isaac Lab.
   - Helped verify task registration, environment loading, reward configuration, observation/action dimensions, and checkpoint paths.
   - Debugged issues related to mismatched task configs, incompatible checkpoints, action dimensions, and command-conditioned policies.

2. **Walking policy development**
   - Developed and refined full-body walking policy configurations.
   - Tuned reward terms for forward velocity, stability, stride length, foot width, arm posture, action smoothness, and termination behavior.
   - Ran and monitored walking policy training experiments, including larger-stride and command-conditioned variants.
   - Generated rollout videos and training/evaluation plots for presentation and final reporting.

3. **Kicking policy development**
   - Adapted walking policies and task designs toward a soccer-style kicking task.
   - Analyzed kicking policy failures, including weak ball contact, ball drift, early falling, and insufficient foot-to-ball interaction.
   - Helped design staged reward curricula emphasizing stable approach, alignment, right-foot approach, contact, and ball-forward motion.
   - Compared our trained checkpoints with teammate-trained checkpoints and identified compatibility issues between 12-DoF and 20-DoF policies.

4. **Experiment analysis and documentation**
   - Interpreted training logs, TensorBoard metrics, reward curves, and qualitative rollout videos.
   - Generated figures for reward trends, termination behavior, and policy diagnostics.
   - Prepared report assets, screenshots, videos, and final project materials.
   - Wrote and organized major parts of the final report and final presentation.

## Summary

Overall, my work focused on making the reinforcement-learning pipeline functional and interpretable, improving walking stability, diagnosing kicking-policy failures, preparing final simulation evidence, and assembling the final submission materials.
