import unittest
import numpy as np
import utils
import os
from tempfile import mkstemp


def test_save_load():
    from pilco.controllers import CombinedController
    max_action = 2.0
    bf = 30
    controller = CombinedController(state_dim=3, control_dim=1, num_basis_functions=bf,
                                    controller_location=np.array([1.0, 0.0, 0.0]), W=np.identity(3),
                                    max_action=max_action)
    _, temp_path = mkstemp()
    utils.save_gpflow_obj_to_path(controller, temp_path)
    c2 = utils.load_controller_from_obj(temp_path)
    os.remove(temp_path)


if __name__ == '__main__':
    test_save_load()
