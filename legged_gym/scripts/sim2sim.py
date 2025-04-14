
# test AMP sim2sim Result 

import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
# from collections import deque
# from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import T1AMPCfg
import torch


class cmd:
    vx = 1.0
    vy = 0.0
    dyaw = 0.0


# def quaternion_to_euler_array(quat):
#     # Ensure quaternion is in the correct format [x, y, z, w]
#     x, y, z, w = quat
    
#     # Roll (x-axis rotation)
#     t0 = +2.0 * (w * x + y * z)
#     t1 = +1.0 - 2.0 * (x * x + y * y)
#     roll_x = np.arctan2(t0, t1)
    
#     # Pitch (y-axis rotation)
#     t2 = +2.0 * (w * y - z * x)
#     t2 = np.clip(t2, -1.0, 1.0)
#     pitch_y = np.arcsin(t2)
    
#     # Yaw (z-axis rotation)
#     t3 = +2.0 * (w * z + x * y)
#     t4 = +1.0 - 2.0 * (y * y + z * z)
#     yaw_z = np.arctan2(t3, t4)
    
#     # Returns roll, pitch, yaw in a NumPy array in radians
#     return np.array([roll_x, pitch_y, yaw_z])

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
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    # default_dof_pos = np.zeros((cfg.env.num_actions), dtype=np.double)
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


    # hist_obs = deque()
    # for _ in range(cfg.env.frame_stack):
    #     hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0


    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([cfg.env.num_observations], dtype=np.float32)
            # obs[0:3] = omega * cfg.normaliaztion.obs_scales.ang_vel
            obs[0:3] = omega * 0.25
            obs[3:6] = gvec 
            # obs[6] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            # obs[7] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            # obs[8] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[6] = cmd.vx * 2
            obs[7] = cmd.vy * 2
            obs[8] = cmd.dyaw * 0.25
            # obs[9:22] = (q - default_dof_pos) * cfg.normalization.obs_scales.dof_pos
            obs[9:22] = (q - default_dof_pos)
            # obs[22:35] = dq * cfg.normalization.obs_scales.dof_vel
            obs[22:35] = dq * 0.05
            obs[35:48] = action

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)


            policy_input = np.zeros([cfg.env.num_observations], dtype=np.float32)
            policy_input = obs
            action[:] = policy(torch.tensor(policy_input)).detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            target_q = action * cfg.control.action_scale


        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q,default_dof_pos,q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau ,-cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    args = parser.parse_args()

    class Sim2simCfg(T1AMPCfg):

        class sim_config:
            # if args.terrain:
            #     mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml'
            # else:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/t1/T1_locomotion.xml'
            sim_duration = 20.0
            dt = 0.001
            decimation = 20

        class robot_config:
            kps = np.array([200, 200, 200, 200, 200, 100, 100, 200, 200, 200, 200, 100,100], dtype=np.double)
            kds = np.array([2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1,1], dtype=np.double)
            tau_limit = np.array([30, 45, 45, 30, 65, 24, 6, 45, 45, 30, 65, 24, 6], dtype=np.double)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
