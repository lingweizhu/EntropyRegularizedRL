import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.drivers import dynamic_step_driver
from tf_agents.networks import q_network
from tf_agents.policies import  random_tf_policy
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from Momentum_DQN import MomentumAgent, ShowProgress


if __name__ == '__main__':
    print(tf.__version__)

    env_name = "CartPole-v0"
    env = suite_gym.load(env_name)

    train_env = tf_py_environment.TFPyEnvironment(env)
    eval_env = tf_py_environment.TFPyEnvironment(env)

    num_iterations = 20000  # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}
    interaction_period = 4

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    fc_layer_params = (100,)
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params
    )

    h_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params
    )

    update_period = 4
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)
    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1.0,
                                                               decay_steps=20000 // update_period,
                                                               end_learning_rate=0.01)
    # beta schedule: look for "On the mixture rate" paragraph of Sec. 5
    beta_fn = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=1.0,
                                                             decay_steps=20000 // update_period,
                                                             decay_rate=0.96)

    agent = MomentumAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        h_network=h_net,
        optimizer=optimizer,
        target_update_period=2000,
        gamma=0.99,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        epsilon_greedy=lambda: epsilon_fn(train_step_counter),
        beta=lambda: beta_fn(train_step_counter)
    )

    agent.initialize()

    eval_policy = agent.policy  # greedy with respect to H
    collect_policy = agent.collect_policy  # epsilon greedy with respect to H

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length
    )

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=4
    )

    # collect some experience before training
    initial_collect_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    init_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch, ShowProgress(2000)],
        num_steps=2000,
    )

    final_time_step, final_policy_state = init_driver.run()



    agent.train = common.function(agent.train)
    driver.run = common.function(driver.run)
    agent.train_step_counter.assign(0)

    dataset = replay_buffer.as_dataset(sample_batch_size=64, num_steps=2, num_parallel_calls=3).prefetch(3)

    time_step = None
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)
    iterator = iter(dataset)

    for iteration in range(num_iterations):
        time_step, policy_state = driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss: {:.5f}".format(iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            print(" average return: ",train_metrics[1].result().numpy())
