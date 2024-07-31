# Pendubot swing-up using Soft Actor-Critic

First execute the setup file, which will create double_pendulum folder, by downloading it from GitHub:
> bash setup.sh

Then, if you want to train the model, run:
> python train.py

This will save the best model, and TensorBoard logs to log_data directory.

Finally, to test the trained agent on the simulation environment, run:
> python evaluate.py

This will create a .mp4 video of the simulation and a .png photo.
