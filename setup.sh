#!/bin/bash

# Clone the repository
git clone http://github.com/turcato-niccolo/double_pendulum.git /kaggle/working/double_pendulum_repo

# We only need this folder
cp -r /kaggle/working/double_pendulum_repo/src/python/double_pendulum .

rm -r -r /kaggle/working/double_pendulum_repo

echo "Operations completed successfully."
