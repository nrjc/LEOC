import unittest
import numpy as np
import unittest
import numpy as np
import utils
import os
from tempfile import mkstemp
from pilco.controllers import CombinedController

from pilco.controllers import RbfController
c2 = utils.load_controller_from_obj('./data/swingup/rbf/swingup_rbf_controller4.pkl')
outc = c2.linearize(np.array([1.,0.,0.]))
print(outc[0][0])
print(outc[0][1])