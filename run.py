from DDPG.ddpg import DDPG, train_agent
from dao.envloader import load_py_env
from dao.trainer import PILCOTrainer, DDPGTrainer
from pilco.controllers import LinearController
from dao.trainer import DDPGTrainer
import dao.envs
import gin
gin.parse_config_file('config.gin')
# #%%
# ddpg_trainer = DDPGTrainer()
# #%%
# ddpg_trainer.train()
#%%
pilco_trainer = PILCOTrainer()
#%%
pilco_trainer.train()
