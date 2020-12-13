# Set metas for the entire experiment/run
import argparse
import gin
import gpflow
import numpy as np
import tensorflow as tf
from dao.trainer import PILCOTrainer, DDPGTrainer

gpflow.config.set_default_float(np.float64)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


parser = argparse.ArgumentParser(description='Run training from configuration file')
parser.add_argument('-file', type=str, help='Location of .gin file containing training config', default='config.gin')
args = parser.parse_args()
gin.parse_config_file(args.file)
try:
    training_type = gin.query_parameter('%type')
except ValueError:
    training_type = "ddpg"
if training_type == "ddpg":
    # %%
    ddpg_trainer = DDPGTrainer()
    # # %%
    ddpg_trainer.train()
elif training_type == "pilco":
    # #%%
    pilco_trainer = PILCOTrainer()
    # %%
    pilco_trainer.train()
else:
    raise NotImplementedError
