import examples.envs
from DDPG.ddpg import DDPG, train_agent
from dao.envloader import load_py_env
from dao.trainer import PILCOTrainer, DDPGTrainer
from pilco.controllers import LinearController

import gin
gin.parse_config_file('config.gin')
#%%
ddpg_trainer = DDPGTrainer()
# pilco_trainer = PILCOTrainer()
# #%%
# pilco_trainer.train()
