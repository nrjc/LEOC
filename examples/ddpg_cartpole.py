import numpy as np
import logging

from tf_agents.environments import suite_gym, tf_py_environment

from DDPG.ddpg import DDPG, ReplayBuffer, train_agent, LinearControllerLayer, MyActorNetwork
from controller_utils import LQR

logging.basicConfig(level=logging.INFO)

np.random.seed(0)

if __name__ == '__main__':
    # Define params
    test_linear_control = False
    num_iterations = 20000  # @param {type:"integer"}
    initial_collect_steps = 1000  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_capacity = 100000  # @param {type:"integer"}
    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 500  # @param {type:"integer"}
    num_eval_episodes = 2  # @param {type:"integer"}
    eval_interval = 500  # @param {type:"integer"}
    target = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

    train_py_env = suite_gym.load('Cartpole-v7')
    eval_py_env = suite_gym.load('Cartpole-v7')
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    A, B, C, Q = train_py_env.control()
    W_matrix = LQR().get_W_matrix(A, B, Q, env='cartpole')

    state_dim = 5
    control_dim = 1

    controller_linear = LinearControllerLayer(state_dim=state_dim, control_dim=control_dim, W=W_matrix)

    if not test_linear_control:

        myDDPGagent = DDPG(train_env, linear_controller=controller_linear, controller_location=target)
        myReplayBuffer = ReplayBuffer(myDDPGagent, train_env, replay_buffer_capacity, initial_collect_steps,
                                      collect_steps_per_iteration)
        train_agent(myDDPGagent,
                    myReplayBuffer,
                    eval_env=eval_env,
                    num_iterations=num_iterations,
                    batch_size=batch_size,
                    initial_collect_steps=initial_collect_steps,
                    collect_steps_per_iteration=collect_steps_per_iteration,
                    num_eval_episodes=num_eval_episodes,
                    log_interval=log_interval,
                    eval_interval=eval_interval)

    else:
        # controller_path = os.path.join(model_save_dir, 'controllers', 'swingup_rbf_controller4.pkl')
        # controller = load_controller_from_obj(controller_path)

        train_py_env.gym.up = True
        train_py_env.reset()

        test_actor = MyActorNetwork(
            train_env.time_step_spec(),
            train_env.action_spec(),
            linear_controller=controller_linear,
            controller_location=target,
            name='ActorNetwork')

        for i in range(batch_size):
            train_env.render()
            timestep = train_env.step(0)
            # state = timestep.observation
            # action = controller_linear(tf.reshape(tf.convert_to_tensor(state), (1, -1)))[0]
            print(f'Step: {i}, timestep: {timestep}')

        test_actor(train_env.timestep)

        train_env.close()
