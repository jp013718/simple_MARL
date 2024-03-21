# Simple MARL
This is a simple example of a MARL problem in which two agents in a gridworld are tasked with reaching each other.
# Setup
In order to begin using this repo, begin by creating a conda environment with Python 3.10.
```
conda create -n MARL python=3.10
conda activate MARL
```
After creating the conda environment, there are several necessary packages, which can be installed using the following commands:
```
python -m pip install ray[rllib]
python -m pip install pygame
conda install -c conda-forge libstdxx-ng
```
# Training
As this environment is incredibly simple, the training does not take long. Training can be performed with
```
python train_PPO.py
```
By default, this trains over 10 epochs and saves a checkpoint every 2. This can be changed by modifying the variables `train_duration` and `save_freq` on lines 40 and 41 of `train_PPO.py`, respectively. Checkpoints are saved to the directory `results`, following the format `algo_[epoch-number]`.
# Viewing Checkpointed Algorithm
In order to watch the agents in the environment, a script is included to load a saved algorithm from a checkpoint and play it in a pygame window. A pre-trained algorithm is included under `results/algo_10` and will be run by default. To run this, simply run the command
```
python player_from_checkpoint.py
```
If you wish to view the performance of an algorithm from a different checkpoint, the variable `checkpoint_dir` can be modified on line 10 of `play_from_checkpoint.py`.