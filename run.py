import examples.envs
from dao.envloader import load_py_env
from dao.trainer import PILCOTrainer, DDPGTrainer
from pilco.controllers import LinearController

import gin
gin.parse_config_file('config.gin')
#%%
pilco_trainer = PILCOTrainer()
#%%
pilco_trainer.train()