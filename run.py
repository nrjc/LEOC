import numpy as np
import gpflow
gpflow.config.set_default_float(np.float32)
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from DDPG.ddpg import DDPG
from DDPG.utils import train_ddpg
from pilco.utils import train_pilco
from dao.envloader import load_py_env
from dao.trainer import PILCOTrainer, DDPGTrainer
from pilco.controllers import LinearController
import dao.envs
import gin


gin.parse_config_file('config.gin')
# %%
# ddpg_trainer = DDPGTrainer()
# # %%
# ddpg_trainer.train()
# #%%
# ddpg_trainer.save()
# #%%
# ddpg_trainer.load()
#%%
pilco_trainer = PILCOTrainer()
#%%
pilco_trainer.train()
#%%
pilco_trainer.save()
#%%
pilco_trainer.load()

result = pilco_trainer.eval()
print(result)
