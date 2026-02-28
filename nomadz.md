# MimicKit -- NomadZ Notes (2026-02-28)

## Where do I start?

MimicKit is a framework to provide motion imitation methods on top of existing reinforcement learning simulators. It helps us to quickly implement and test new motion imitation methods without having to implement the underlying details of the reinforcement learning algorithms. 

For an overview of how this framework works, refer to [Starter Guide](https://arxiv.org/pdf/2510.13794) by Xue Bin Peng, the author of this framework. In essence, we are able to tweak the environments created for DeepMimic, AMP, ASE and ADD for our own use. For NomadZ, the main algorithms we will likely use are [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html) and [AMP](https://xbpeng.github.io/projects/AMP/index.html). It is highly recommended for you to read these papers, as recent motion imitation methods are often based on these two algorithms. 

## What has been done?

### Video to Humanoid Motion Pipeline

[nomadz-ethz/motion_data_generation](https://github.com/nomadz-ethz/motion_data_generation) puts together [GVHMR](https://github.com/zju3dv/GVHMR) (which generates gravity-aligned human mesh recovery from video) and [GMR](https://github.com/nomadz-ethz/GMR) (which retargets the motion from human mesh to humanoid skeleton) to create a pipeline that can generate humanoid motion data from video. I highly recommend reading the original papers for [GVHMR](https://zju3dv.github.io/gvhmr/) and [GMR](https://jaraujo98.github.io/retargeting_matters/) too. The resulting pkl file can be converted to MimicKit format using the script [gmr_to_mimickit.py](tools/gmr_to_mimickit/gmr_to_mimickit.py). 

### Support for Booster K1 Robot

The original MimicKit library does not have support for Booster K1 robot. Based on their implementation for the Unitree G1, I have added support for Booster K1 robot by adding its [assets](data/assets/k1), then also created DeepMimic [agents](data/agents/deepmimic_k1_ppo_agent.yaml) and [envs](data/envs/deepmimic_k1_env.yaml) as examples. You can run the example environment using the following command:

```bash
python mimickit/run.py --mode train --num_envs 2 --engine_config data/engines/isaac_lab_engine.yaml --env_config data/envs/deepmimic_k1_env.yaml --agent_config data/agents/deepmimic_k1_ppo_agent.yaml --visualize true --out_dir output/
```

The assets for K1 were taken from [booster_assets](https://github.com/BoosterRobotics/booster_assets/tree/main/robots/K1) and [K1 Instruction Manual](https://booster.feishu.cn/wiki/E3q5wF5SnitXZgkY18Uc8odBnXb), then adjusted to work with Isaac and MimicKit (mainly changing link/joint names). 

## What needs to be done?

### Setup on NomadZ Workstation

I have performed all experiments on my personal laptop (Ubuntu 24.04). The NomadZ workstation needs to be setup for MimicKit to work. On my personal laptop, I have installed Isaac Sim (5.1.0) and Isaac Lab (2.3.0) using uv pip ([Instruction](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)). The workstation already has an installation of Isaac Sim and Isaac Lab of the right versions, so we can link it to a new conda environment and install MimicKit there. I have not done so because of no disk space on the workstation, so once we have it backed-up and freshly partitioned, we can set it up for MimicKit. 

The assets and motion data for K1 are not tracked on this repository. I am unsure of the license / copyright rules for these assets, so I have not added them to this repository. We can either confirm that this is okay, or add it to our private drive for now. 

### Creating SLURM script for Euler Cluster Training

Ideally, we have SLURM scripts to run training on the Euler cluster to avoid over-loading the workstation. A common workflow is to create a Singularity image of Isaac Sim + Isaac Lab + dependencies installed, then connect it to the MimicKit scripts. This would take around 20-30GB, which can be done on your home directory on the Euler cluster if you set it up. We can explore this together once we have more environments working for training. 

### Obtaining Clean Motion Data

DeepMimic and AMP use an early termination algorithm to end the episode when specific conditions are met (e.g. the humanoid falls down, goes out of tracked motion, etc.) to accelerate learning. When you run the example DeepMimic environment for K1, you will see that this early termination is triggered at a very high frequency. 

The main suspect for this problem is our motion data; because we have noisy data (generated from video with our GVHMR+GMR pipeline), we have ground foot penetration and other artifacts that cause the K1 to trigger early termination. This would extremely slow down the training or even fail to train at all. We have a few options to solve this problem:

1. Create a post-processing method to GVHMR+GMR pipeline to clean up the motion data. 
2. Use an alternate pipeline with ground foot penetration constraint. 
3. Use clean motion data from other sources (e.g. motion capture data such as [AMASS](https://mocap.cs.cmu.edu/search.php?maincat=4&subcat=7)).

The first and second options are beneficial for long-term research and usability. It is an ongoing field of research (related to my semester project), and would be a great contribution to the RoboCup research. The third option is probably the way we want to start out -- it also provides plenty of data for walking, running, rotating, jumping, etc. that would be useful for our K1. 

Whichever method we end up choosing, we can add the resulting motion data to [motions](data/motions/k1) then use it for training any of the algorithms. 

### Implementing AMP and Dynamic Ball Environment

The DeepMimic example for K1 only learns to imitate the motion data exactly. A new AMP environment would allow us to achieve both natural motion (via the style reward) and task performance (via the task reward, e.g. steering or kicking a ball towards a target). There is an example AMP environment for the steering task in [task_steering_env.py](mimickit/envs/task_steering_env.py). We can use this as a reference to create a new AMP environment for K1. The benefit of MimicKit is that it is a very thin layer on top of Isaac Lab -- we can flexibly add a dynamic object like a ball with similar code to how it is done in Isaac Lab, then adjust the rewards and observations as needed. 

It may be helpful to look at this env hierarchy diagram to understand how the environments are made in MimicKit:

```bash
BaseEnv (base_env.py)            — Abstract: defines reset(), step(), obs/action spaces
  └── SimEnv (sim_env.py)        — Abstract: adds physics engine, sim tensors, rendering loop
        └── CharEnv (char_env.py) — Concrete: loads a character, PD control, basic obs/reward
              ├── DeepMimicEnv (deepmimic_env.py)  — Adds motion tracking (reference motions, imitation reward)
              │     ├── AMPEnv (amp_env.py)         — Replaces tracking reward with discriminator observations
              │     │     ├── ASEEnv (ase_env.py)    — Adds latent skill embeddings on top of AMP
              │     │     ├── ADDEnv (add_env.py)    — Adds demo discriminator observations on top of AMP
              │     │     ├── TaskLocationEnv (task_location_env.py)  — AMP + go-to-target task
              │     │     └── TaskSteeringEnv (task_steering_env.py)  — AMP + velocity/heading task
              │     └── StaticObjectsEnv (static_objects_env.py)     — DeepMimic + static rigid objects
              ├── ViewMotionEnv (view_motion_env.py)   — Replays motions (no RL, visualization only)
              └── CharDofTestEnv (char_dof_test_env.py) — Cycles through DOFs (debugging/testing only)
```

## What are our next steps?

I suggest the following immediate next steps:

- Set up the workstation for MimicKit and create a SLURM script for training on the Euler cluster. 
- Create a conversion script from AMASS motion data to MimicKit format for K1 motion. Use it to test the example DeepMimic environment for K1 and see if we can get it to train without triggering early termination.
- Create a new AMP environment for Unitree G1 to kick or dribble a ball. If it works, we can convert it to K1 and test it out. 

In the long term, we may investigate into the following:

- Create a retargeting pipeline to go from video to clean K1 motion data without foot penetration or artifacts. 
- Experiment with the use of ASE / ADD algorithms. ASE in theory would be very useful for learning a latent space of skills which would be perfect for soccer, but not much literature work is done like DeepMimic and AMP. 
- Come up with novel contributions to motion imitation algorithms for RoboCup research. Applying new papers like [BeyondMimic](https://arxiv.org/pdf/2508.08241)? Generating a large dataset of already-labeled and clean data from FIFA video games? Showcasing more dynamic kicks (overhead-kick, bicycle-kick, etc.) that are hard to learn? 