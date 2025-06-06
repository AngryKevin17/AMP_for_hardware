# Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions #

Codebase for the "[Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions](https://bit.ly/3hpvbD6)" project. This repository contains the code necessary to ground agent skills using small amounts of reference data (4.5 seconds). All experiments are performed using the A1 robot from Unitree. This repository is based off of Nikita Rudin's [legged_gym](https://github.com/leggedrobotics/legged_gym) repo, and enables us to train policies using [Isaac Gym](https://developer.nvidia.com/isaac-gym).

**Maintainer**: Alejandro Escontrela
**Affiliation**: University of California at Berkeley
**Contact**: escontrela@berkeley.edu

### Useful Links ###
Project website: https://bit.ly/3hpvbD6
Paper: https://drive.google.com/file/d/1kFm79nMmrc0ZIiH0XO8_HV-fj73agheO/view?usp=sharing

### Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended). i.e. with conda:
    - `conda create -n amp_hw python==3.8`
    - `conda activate amp_hw`
2. Install pytorch with cuda:
    - `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
    - `pip install pybullet==3.2.1 opencv-python==4.5.5.64 tensorboard mujoco mujoco-python-viewer tqdm wandb`
    - `conda install numpy=1.23`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   - Clone this repository
   -  `cd AMP_for_hardware/rsl_rl && pip install -e .` 
5. Install legged_gym
   - `cd ../ && pip install -e .`
6. Configure the environment to handle shared libraries
    ```sh
    $ cd $CONDA_PREFIX
    $ mkdir -p ./etc/conda/activate.d
    $ vim ./etc/conda/activate.d/env_vars.sh  # Add the following line
    export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
    $ mkdir -p ./etc/conda/deactivate.d
    $ vim ./etc/conda/deactivate.d/env_vars.sh  # Add the following line
    export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
    unset OLD_LD_LIBRARY_PATH
    ```

### CODE STRUCTURE ###
1. Each environment is defined by an env file (`legged_robot.py`) and a config file (`legged_robot_config.py`). The config file contains two classes: one conatianing all the environment parameters (`LeggedRobotCfg`) and one for the training parameters (`LeggedRobotCfgPPo`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward. The AMP reward parameters are defined in `LeggedRobotCfgPPO`, as well as the path to the reference data.
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.
5. Reference data can be found in the `datasets` folder.

### Usage ###
1. Train:  
  ```python legged_gym/scripts/train.py --task=t1_amp``
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `AMP_for_hardware/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    -  The following command line arguments override the values set in the config files:
     - --task TASK: Task name.
     - --resume:   Resume training from a checkpoint
     - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
     - --run_name RUN_NAME:  Name of the run.
     - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
     - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - --num_envs NUM_ENVS:  Number of environments to create.
     - --seed SEED:  Random seed.
     - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
2. Play a trained policy:  
```python legged_gym/scripts/play.py --task=t1_amp```
    - By default the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.
3. Record video of a trained policy
```python legged_gym/scripts/record_policy.py --task=t1_amp```
    - This saves a video of the in the base directory.
4. Sim2Sim test in Mujoco
```python legged_gym/scripts/sim2sim.py --task=t1_amp --load_model=PATH_TO_EXPORTED_JIT_MODEL```
    - PATH_TO_EXPORTED_JIT_MODEL: LEGGED_GYM_ROOT_DIR/logs/t1_amp/exported/policies/policy_1.pt

### Adding a new environment ###
The base environment `legged_robot` implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and no reward scales. 

1. Add a new folder to `envs/` with `'<your_env>_config.py`, which inherit from an existing environment cfgs  
2. If adding a new robot:
    - Add the corresponding assets to `resourses/`.
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
3. (If needed) implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
4. Register your env in `isaacgym_anymal/envs/__init__.py`.
5. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!


### Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `sudo apt install libpython3.8`

### Known Issues ###
1. The contact forces reported by `net_contact_force_tensor` are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors, but the force are propagated through the sensors of consecutive bodies resulting in an undesireable behaviour. However, for a legged robot it is possible to add sensors to the feet/end effector only and get the expected results. When using the force sensors make sure to exclude gravity from trhe reported forces with `sensor_options.enable_forward_dynamics_forces`. Example:
```
    sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False # for example gravity
        sensor_options.enable_constraint_solver_forces = True # for example contacts
        sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.
```
