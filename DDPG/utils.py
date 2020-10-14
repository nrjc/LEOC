import gin
import numpy as np
from tf_agents.utils import common


@gin.configurable
def train_ddpg(ddpg, replay_buffer, eval_env, num_iterations, batch_size=64, initial_collect_steps=1000,
               collect_steps_per_iteration=1, num_eval_episodes=5, log_interval=200, eval_interval=1000):
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    ddpg.agent.train = common.function(ddpg.agent.train)

    # Reset the train step
    ddpg.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = ddpg.compute_avg_return(eval_env, num_episodes=num_eval_episodes)
    returns = [avg_return]
    lambdas = [ddpg.actor_network.S.numpy()]
    best_return = -np.finfo(np.float32).max

    # Collect some initial experience
    replay_buffer.collect_data(steps=initial_collect_steps)

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        replay_buffer.collect_data(steps=collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        iterator = replay_buffer.get_dataset(batch_size)
        experience, _ = next(iterator)
        train_loss = ddpg.agent.train(experience).loss

        step = ddpg.train_step_counter.numpy()
        lambdas.append(ddpg.actor_network.S.numpy())

        # if step % log_interval == 0:
        #     print(f'step = {step}: loss = {train_loss}')
        #     # print(f'S: {ddpg.actor_network.S}, r: {ddpg.actor_network.r}')

        if step % eval_interval == 0:
            avg_return = ddpg.compute_avg_return(eval_env, num_episodes=num_eval_episodes)
            print(f'step = {step}: Average Return = {avg_return}')
            returns.append(avg_return)
            # utils_plot(num_iterations, eval_interval, returns)

            # if step > int(num_iterations / 5 * 4) and avg_return > best_return:
            #     best_return = avg_return
            #     ddpg.policy_saver.save(f'policy_{step // eval_interval}')

    print(f'returns {returns}')
    # print(f'lambdas {lambdas}')
    print(f'Finished training for {num_iterations} iterations')
    return ddpg