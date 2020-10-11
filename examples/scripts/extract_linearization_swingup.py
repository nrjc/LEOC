import numpy as np
from pilco import utils

c2 = utils.load_controller_from_obj('./data/swingup/rbf/swingup_rbf_controller4.pkl')
outc = c2.linearize(np.array([1.,0.,0.]))
print(outc[0][0])
print(outc[0][1])