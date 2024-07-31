# A file containing the code for training the RL agent

import os
import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func, CustomEnv
from double_pendulum.utils.wrap_angles import wrap_angles_diff

seed = 0
random.seed(seed)
np.random.seed(seed)

# parameters
robot = 'pendubot'  # pendubot = the shoulder is active and elbow is passive
torque_limit = [10.0, 0.0]
active_act = 0

# simulation parameters
integrator = 'runge_kutta'
max_velocity = 20.0  # default value
goal = [np.pi, 0.0, 0.0, 0.0]

log_dir = os.path.join(os.getcwd(), 'log_data', robot + '_training')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

model_par_path = os.path.join(os.getcwd(), 'parameters', robot + '_parameters.yml')
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# Initialize plant and simulator
plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)

############################################################################
# Learning environment parameters
state_representation = 2  # default value
obs_space = gym.spaces.Box(np.array([-1.0]*4), np.array([1.0]*4), dtype=np.float32)
act_space = gym.spaces.Box(np.array([-1.0]), np.array([1.0]), dtype=np.float32)
############################################################################
#
############################################################################
# Tuning parameters
dt = 0.01  # 100Hz
max_steps = 500  # 5s per episode at 100Hz
n_envs = 100
training_steps = int(3e7)  # int(1e6) default
verbose = 1
reward_threshold = 1e10
eval_freq = 2500
n_eval_episodes = 10
learning_rate = 0.01  # 0.0003 default
############################################################################
#
############################################################################
# Initialize dynamics function
dynamics_func = double_pendulum_dynamics_func(simulator=simulator,
                                              dt=dt,
                                              integrator=integrator,
                                              robot=robot,
                                              state_representation=state_representation,
                                              max_velocity=max_velocity,
                                              torque_limit=torque_limit)
############################################################################
#
############################################################################
#  Define reward function

# LQR parameters -> for termination
# source: https://github.com/turcato-niccolo/double_pendulum/blob/main/leaderboard/pendubot/simulation_v1/con_sac_lqr.py
S = np.array([
    [7.857934201124567153e01, 5.653751913776947191e01, 1.789996146741196981e01, 8.073612858295813766e00],
    [5.653751913776947191e01, 4.362786774581156379e01, 1.306971194928728330e01, 6.041705515910111401e00],
    [1.789996146741196981e01, 1.306971194928728330e01, 4.125964000971944046e00, 1.864116086667296113e00],
    [8.073612858295813766e00, 6.041705515910111401e00, 1.864116086667296113e00, 8.609202333737846491e-01]
])
rho = 1.690673829091186575e-01

#  The following are global variables for the reward function
l1 = mpar.l[0]  # 0.4
l2 = mpar.l[1]  # 0.1

hline_max = l1 + l2  # 0.5
hline_1 = 0.6 * hline_max  # 0.3
hline_2 = 0.8 * hline_max  # 0.4
hline_3 = 0.9 * hline_max  # 0.45

Q = np.diag([10.0, 10.0, 0.1, 0.1])
R = np.array([[1e-4]])

def get_coordinates_of_pend_tips(theta1, theta2):
    # From SymbolicDoublePendulum.forward_kinematics()
    x1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.sin(theta1 + theta2)
    y1 = -l1 * np.cos(theta1)
    y2 = y1 - l2 * np.cos(theta1 + theta2)
    return x1, y1, x2, y2

def reward_func(observation, action):
    # Transform the normalized space used by the learning algorithm to the actual physical state space of the pendulum.
    # state
    s = np.array(
        [
            (observation[0] * np.pi + np.pi),  # [0, 2pi]
            (observation[1] * np.pi + 2 * np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2] * dynamics_func.max_velocity,
            observation[3] * dynamics_func.max_velocity,
        ]
    )
    u = torque_limit[active_act] * action
    x1, y1, x2, y2 = get_coordinates_of_pend_tips(theta1=s[0], theta2=s[1])
    delta = np.subtract(s, goal)
    quadratic_cost = np.einsum("i, ij, j", delta, Q, delta) + np.einsum("i, ij, j", u, R, u)
    reward = -quadratic_cost
    if y1 > 0.8 * l1 and y2 > y1:
        height_bonus = y2 / hline_max  # If we are above a certain line, we get bonus
        reward += height_bonus * 1e3  # Add bonus for height
        velocity_penalty = max(0, abs(s[2]) + abs(s[3]) - 8.0)
        reward -= velocity_penalty * 1e2
        if np.einsum("i, ij, j", delta, S, delta) < rho:  # Bonus for reaching ROA
            reward += 1e4
    return reward

def terminated_func(observation):
    s = np.array(
        [
            (observation[0] * np.pi + np.pi),
            (observation[1] * np.pi + 2 * np.pi) % (2 * np.pi) - np.pi,
            observation[2] * dynamics_func.max_velocity,
            observation[3] * dynamics_func.max_velocity,
        ]
    )
    delta = np.subtract(s, goal)
    if np.einsum("i, ij, j", delta, S, delta) < rho:  # ROA reached
        return True
    return False

def noisy_reset_func():
    rand = np.array(np.random.rand(4) * 0.01, dtype=np.float32)
    observation = np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32) + rand
    return observation

def zero_reset_func():
    observation = [-1.0, -1.0, 0.0, 0.0]
    return observation


# Initialize vectorized environment
env = CustomEnv(dynamics_func=dynamics_func,
                reward_func=reward_func,
                terminated_func=terminated_func,
                reset_func=noisy_reset_func,
                obs_space=obs_space,
                act_space=act_space,
                max_episode_steps=max_steps)

# Training env
envs = make_vec_env(env_id=CustomEnv,
                    n_envs=n_envs,
                    seed=seed,
                    env_kwargs={"dynamics_func": dynamics_func,
                                "reward_func": reward_func,
                                "terminated_func": terminated_func,
                                "reset_func": noisy_reset_func,
                                "obs_space": obs_space,
                                "act_space": act_space,
                                "max_episode_steps": max_steps})

# Evaluation env
eval_env = CustomEnv(dynamics_func=dynamics_func,
                     reward_func=reward_func,
                     terminated_func=terminated_func,
                     reset_func=zero_reset_func,
                     obs_space=obs_space,
                     act_space=act_space,
                     max_episode_steps=max_steps)

# Training callbacks
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=verbose)

eval_callback = EvalCallback(eval_env,
                             callback_on_new_best=callback_on_best,
                             best_model_save_path=os.path.join(log_dir, "best_model"),
                             log_path=log_dir,
                             eval_freq=eval_freq,
                             verbose=verbose,
                             n_eval_episodes=n_eval_episodes)

# Train
agent = SAC(MlpPolicy,
            envs,
            verbose=verbose,
            tensorboard_log=os.path.join(log_dir, "tb_logs"),
            learning_rate=learning_rate,
            # gamma=0.99,
            seed=seed,
            device='auto',
            action_noise=NormalActionNoise(mean=[0.0], sigma=[0.1]))   # Noise is added for exploration

agent.learn(total_timesteps=training_steps, callback=eval_callback, progress_bar=False)
