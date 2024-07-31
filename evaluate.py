# A file that loads the model weights from memory, and evaluates such agent on the environment

import os
import numpy as np
import shutil

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.utils.wrap_angles import wrap_angles_diff
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.analysis.leaderboard import get_swingup_time

# LQR parameters
Q_lqr = np.diag([1.92, 1.92, 0.3, 0.3])
R_lqr = np.diag([0.82, 0.82])
S = np.array([
    [7.857934201124567153e01, 5.653751913776947191e01, 1.789996146741196981e01, 8.073612858295813766e00],
    [5.653751913776947191e01, 4.362786774581156379e01, 1.306971194928728330e01, 6.041705515910111401e00],
    [1.789996146741196981e01, 1.306971194928728330e01, 4.125964000971944046e00, 1.864116086667296113e00],
    [8.073612858295813766e00, 6.041705515910111401e00, 1.864116086667296113e00, 8.609202333737846491e-01]
])
rho = 8.690673829091186575e-01  # 1.690673829091186575e-01

# Controllers switching conditions:
def condition1(t, x):
    delta = wrap_angles_diff(np.subtract(x, goal))
    return np.einsum("i,ij,j", delta, S, delta) > 1.1 * rho

def condition2(t, x):
    delta = wrap_angles_diff(np.subtract(x, goal))
    return np.einsum("i,ij,j", delta, S, delta) < rho  # switch controller


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
mpar = model_parameters(filepath=os.path.join(os.getcwd(), 'parameters', robot + '_parameters.yml'))
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# Simulation parameters
dt = 0.001  # 1kHz
t_final = 5.0  # Try also with 10.0
integrator = 'runge_kutta'
goal = [np.pi, 0.0, 0.0, 0.0]

# Plant and Simulator
plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

# Controller 1 = SAC
model_path = os.path.join(os.getcwd(), 'log_data',  robot + '_training', 'best_model', 'best_model.zip')
dynamics_func = double_pendulum_dynamics_func(simulator=sim,
                                              dt=dt,
                                              integrator=integrator,
                                              robot=robot,
                                              torque_limit=torque_limit,
                                              max_velocity=20.0,  # default
                                              state_representation=2,  # default
                                              scaling=True)  # default
controller1 = SACController(model_path=model_path,
                            dynamics_func=dynamics_func,
                            dt=dt,
                            scaling=True)  # default

# Controller 2 = LQR -> for stabilization
controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q_lqr, R=R_lqr)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=15)

# Initialize combined controller
controller = CombinedController(controller1=controller1,
                                controller2=controller2,
                                condition1=condition1,
                                condition2=condition2,
                                compute_both=False)  # default 
controller.init()

# Start simulation
T, X, U = sim.simulate_and_animate(t0=0.0,
                                   x0=[0.0]*4,
                                   tf=t_final,
                                   dt=dt,
                                   controller=controller,
                                   integrator=integrator,
                                   save_video=True,
                                   video_name=robot+'.mp4')

print('Swingup time: '+str(get_swingup_time(T, np.array(X), mpar=mpar)))
# Swingup time: 1.204999999999978

# Plot timeseries
plot_timeseries(T, X, U,
                X_meas=sim.meas_x_values,
                pos_y_lines=[np.pi],  # [0.0, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                save_to=robot+'.png')

# Move the video and plot to results folder
destination_folder = os.path.join(os.getcwd(), 'results')
os.makedirs(destination_folder, exist_ok=True)
source_video = os.path.join(os.getcwd(), robot+'.mp4')
source_plot = os.path.join(os.getcwd(), robot+'.png')
destination_video = os.path.join(destination_folder, robot+'.mp4')
destination_plot = os.path.join(destination_folder, robot+'.png')

shutil.move(source_video, destination_video)
shutil.move(source_plot, destination_plot)
