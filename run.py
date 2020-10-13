from DDPG.ddpg import DDPG, train_ddpg
from pilco.utils import train_pilco
from dao.envloader import load_py_env
from dao.trainer import PILCOTrainer, DDPGTrainer
from pilco.controllers import LinearController
import dao.envs
import gin
gin.parse_config_file('config.gin')
#%%
ddpg_trainer = DDPGTrainer()
#%%
# ddpg_trainer.train()
#%%
ddpg_trainer.save()
#%%
ddpg_trainer.load()
#%%
pilco_trainer = PILCOTrainer()
#%%
pilco_trainer.train()
#%%
pilco_trainer.save()
#%%
pilco_trainer.load()

