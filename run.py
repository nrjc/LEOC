# Set metas for the entire experiment/run
import numpy as np
import gpflow

from dao.plotter import EpochPlotter

gpflow.config.set_default_float(np.float64)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gin

# Import relevant local modules
import dao.envs
from dao.envloader import load_py_env
from dao.trainer import PILCOTrainer, DDPGTrainer


gin.parse_config_file('config.gin')
# %%
# ddpg_trainer = DDPGTrainer()
# # %%
# ddpg_trainer.train()
# #%%
# ddpg_trainer.save()
# #%%
# ddpg_trainer.load()
# #%%
# result = ddpg_trainer.eval()
# print(result)
#%%
pilco_trainer = PILCOTrainer()
#%%
# pilco_trainer.train()
# #%%
# pilco_trainer.save()
#%%
pilco_trainer.load()

# pilco_trainer.visualise()

trajectories = pilco_trainer.eval()

myEpochPlotter = EpochPlotter(trajectories)
myEpochPlotter.plot_state_cum_ratio()
