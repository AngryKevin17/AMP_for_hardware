
# test AMP sim2sim Result 

import os
import sys
import math
import time
import numpy as np
import mujoco
from mujoco.viewer import launch_passive
from tqdm import tqdm
import torch

LEGGED_GYM_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))



class cmd:
    vx = 1.0
    vy = 0.0
    dyaw = 0.0

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
    return (q, dq, quat, omega, gvec)

def pd_control(target_q, default_q ,q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q + default_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    viewer = launch_passive(model, data)

    num_dofs = cfg.env.num_actions + 2
    target_q = np.zeros((num_dofs), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    default_dof_pos = np.array([
        0.0,        # Waist
        -0.2,       # Left_Hip_Pitch
        0.0,        # Left_Hip_Roll
        0.0,        # Left_Hip_Yaw
        0.4,        # Left_Knee_Pitch
        -0.25,      # Left_Ankle_Pitch
        0.0,        # Left_Ankle_Roll
        -0.2,       # Right_Hip_Pitch
        0.0,        # Right_Hip_Roll
        0.0,        # Right_Hip_Yaw
        0.4,        # Right_Knee_Pitch
        -0.25,      # Right_Ankle_Pitch
        0.0         # Right_Ankle_Roll
    ], dtype=np.double)

    count_lowlevel = 0
    stacked_obs = np.zeros((cfg.env.include_history_steps, cfg.env.num_observations), dtype=np.float32)


    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, omega, gvec = get_obs(data)
        q = q[-num_dofs:]
        dq = dq[-num_dofs:]

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([cfg.env.num_observations], dtype=np.float32)
            obs[0:3] = omega * 0.25
            obs[3:6] = gvec 
            obs[6] = cmd.vx * 2
            obs[7] = cmd.vy * 2
            obs[8] = cmd.dyaw * 0.25
            obs[9:22] = (q - default_dof_pos)
            obs[22:35] = dq * 0.05
            obs[35:46] = action

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            if count_lowlevel == 0:
                stacked_obs = np.tile(obs, (cfg.env.include_history_steps, 1))
            else:
                stacked_obs[:-1] = stacked_obs[1:]
                stacked_obs[-1] = obs

            action[:] = policy({
                "observations": torch.tensor(obs),
                "encoder_observations": torch.tensor(stacked_obs.flatten())
            }).detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            target_q[cfg.control.effective_ids] = action * cfg.control.action_scale


        target_dq = np.zeros((num_dofs), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q,default_dof_pos,q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau ,-cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.cam.lookat[:] = data.qpos.astype(np.float32)[0:3]
        viewer.sync()
        time.sleep(cfg.sim_config.dt)
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    args = parser.parse_args()

    class Sim2simCfg:

        class env:
            num_observations = 46
            num_privileged_obs = 49
            num_actions = 11
            include_history_steps = 50

        class control:
            action_scale = 0.25
            effective_ids = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11] # ids of the joints that are controlled by the policy

        class normalization:
            class obs_scales:
                lin_vel = 2.0
                ang_vel = 0.25
                dof_pos = 1.0
                dof_vel = 0.05
                height_measurements = 5.0
            clip_observations = 18.
            clip_actions = 4.

        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/t1/T1_locomotion.xml'
            sim_duration = 20.0
            dt = 0.002
            decimation = 10

        class robot_config:
            kps = np.array([200, 200, 200, 200, 200, 100, 100, 200, 200, 200, 200, 100,100], dtype=np.double)
            kds = np.array([2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1,1], dtype=np.double)
            tau_limit = np.array([30, 45, 45, 30, 65, 24, 6, 45, 45, 30, 65, 24, 6], dtype=np.double)


if __name__ == "__main__":
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
