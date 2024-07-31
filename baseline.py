# A file that runs the baseline / heuristic policy on the environment

import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.utils.wrap_angles import wrap_angles_diff
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.analysis.leaderboard import get_swingup_time

seed = 0

# LQR parameters
Q_lqr = np.diag([1.92, 1.92, 0.3, 0.3])
R_lqr = np.diag([0.82, 0.82])
S = np.array([
    [7.857934201124567153e01, 5.653751913776947191e01, 1.789996146741196981e01, 8.073612858295813766e00],
    [5.653751913776947191e01, 4.362786774581156379e01, 1.306971194928728330e01, 6.041705515910111401e00],
    [1.789996146741196981e01, 1.306971194928728330e01, 4.125964000971944046e00, 1.864116086667296113e00],
    [8.073612858295813766e00, 6.041705515910111401e00, 1.864116086667296113e00, 8.609202333737846491e-01]
])
# rho = 1.690673829091186575e-01
rho = 8.690673829091186575e-01

# Controllers switching conditions:

def condition1(t, x):
    delta = wrap_angles_diff(np.subtract(x, goal))
    if np.einsum("i,ij,j", delta, S, delta) > 1.1 * rho:
        return True
    return False

def condition2(t, x):
    delta = wrap_angles_diff(np.subtract(x, goal))
    if np.einsum("i,ij,j", delta, S, delta) < rho:  # switch controller
        return True
    return False


# Model parameters
robot = 'pendubot'
if robot == 'pendubot':  # pendubot = the shoulder is active and elbow is passive
    torque_limit = [10.0, 0.0]
    active_act = 0
elif robot == 'acrobot':
    torque_limit = [0.0, 10.0]
    active_act = 1
else:
    raise ValueError(f'Invalid robot value entered: {robot}. Please choose among: pendubot/acrobot.')

# model for simulation / controller
model_par_path = os.path.join(os.getcwd(), 'parameters', robot + '_parameters.yml')
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# Simulation parameters
# We will use the dt value used for training the SAC agent in order to calculate the return that the baseline obtains:
dt = 0.01  # 100Hz
t_final = 5.0  # 5s

integrator = 'runge_kutta'
goal = [np.pi, 0.0, 0.0, 0.0]
max_velocity = 20.0

# Plant
plant = SymbolicDoublePendulum(model_pars=mpar)
# Simulator
sim = Simulator(plant=plant)

# Controller 1

# class DummyController(AbstractController):
#     def get_control_output_(self, x, t=None):
#         return torque_limit

# class ChargingController(AbstractController):
#     def get_control_output_(self, x, t=None):
#         if t < 0.2:  # charge time
#             return torque_limit
#         else:
#             u = [0.0, 0.0]
#             u[active_act] = -torque_limit[active_act]
#             return u

# class EnergyBasedController(AbstractController):
#     def __init__(self):
#         super().__init__()
#         self.desired_energy = plant.total_energy(goal)

#     def get_control_output_(self, x, t=None):
#         if t < dt:  # initial_push_duration
#             return torque_limit
#         else:
#             x = wrap_angles_diff(x)
#             theta1, theta2, omega1, omega2 = x
#             current_energy = plant.total_energy(x)
#             energy_error = self.desired_energy - current_energy
#             # Apply maximum torque in the direction that increases energy
#             u = energy_error * np.sign(omega1)
#             u = np.clip(u, -torque_limit[active_act], torque_limit[active_act])
#             return [u, 0.0]

controller1 = PointPIDController(torque_limit=torque_limit, dt=dt)
controller1.set_parameters(Kp=6.1, Kd=0.2, Ki=0.1)  # Use this setup for dt = 0.01
# For higher refresh rate we need to update the parameters (we use dt = 0.001 for testing the SAC agent)

# Controller 2 = LQR -> for stabilization
controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q_lqr, R=R_lqr)
controller2.set_parameters(failure_value=0., cost_to_go_cut=100)

# Initialize combined controller
controller = CombinedController(controller1=controller1,
                                controller2=controller2,
                                condition1=condition1,
                                condition2=condition2,
                                compute_both=False)
controller.init()

print('Running simulation.')
# Start simulation
T, X, U = sim.simulate_and_animate(t0=0.0,
                                   x0=[0.0]*4,
                                   tf=t_final,
                                   dt=dt,
                                   controller=controller,
                                   integrator=integrator,
                                   save_video=True,
                                   video_name=robot+'_baseline.mp4')

print('Swingup time: '+str(get_swingup_time(T, np.array(X), mpar=mpar)))

print('Plot timeseries.')
# Plot timeseries
plot_timeseries(T, X, U,
                X_meas=sim.meas_x_values,
                pos_y_lines=[np.pi],  # [0.0, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                save_to=robot+'_baseline.png')

# Move the video and plot to results folder
destination_folder = os.path.join(os.getcwd(), 'results')
os.makedirs(destination_folder, exist_ok=True)
source_video = os.path.join(os.getcwd(), robot+'_baseline.mp4')
source_plot = os.path.join(os.getcwd(), robot+'_baseline.png')
destination_video = os.path.join(destination_folder, robot+'_baseline.mp4')
destination_plot = os.path.join(destination_folder, robot+'_baseline.png')

shutil.move(source_video, destination_video)
shutil.move(source_plot, destination_plot)

# This code is needed for computing the episode return.

l1 = mpar.l[0]  # 0.4
l2 = mpar.l[1]  # 0.1
hline_max = l1 + l2  # 0.5
Q = np.diag([10.0, 10.0, 0.1, 0.1])
R = np.array([[1e-4]])

def get_coordinates_of_pend_tips(theta1, theta2):
    x1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.sin(theta1 + theta2)
    y1 = -l1 * np.cos(theta1)
    y2 = y1 - l2 * np.cos(theta1 + theta2)
    return x1, y1, x2, y2

def reward_func(observation, action):
    s = wrap_angles_diff(observation)
    delta = np.subtract(s, goal)
    u = np.array([torque_limit[active_act] * action[active_act]])
    x1, y1, x2, y2 = get_coordinates_of_pend_tips(s[0], s[1])
    quadratic_cost = np.einsum("i, ij, j", delta, Q, delta) + np.einsum("i, ij, j", u, R, u)
    reward = -quadratic_cost
    if y1 > 0.8 * l1 and y2 > y1:
        height_bonus = y2 / hline_max  # if we are above a certain line, we get bonus
        reward += height_bonus * 1e3  # Add bonus for height
        velocity_penalty = max(0, (abs(s[2]) + abs(s[3])) - 8.0)
        reward -= velocity_penalty * 1e2
        if np.einsum("i, ij, j", delta, S, delta) < rho:  # Bonus for reaching ROA
            reward += 1e4
    return reward


print('Calculating reward.')
rewards = []

for _t, _x, _u in zip(T, X, U):  # Compute rewards for the episode
    rew = reward_func(_x, _u)
    rewards.append(rew)

ep_rew_total = np.sum(rewards)
print(f"Reward for the episode: {ep_rew_total:.2f}")

# Swingup time: 4.759999999999943
# Reward for the episode: 103320.86
