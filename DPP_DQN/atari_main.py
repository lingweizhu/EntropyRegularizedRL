from Munchausen_DQN import *
# from DPP_DQN import *
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from DPP_DQN import *
# from DPP_DQN import *
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy


if __name__ == '__main__':
    print(tf.__version__)

    env_name = "BreakoutNoFrameskip-v4"

    max_episode_steps = 27000
    env = suite_atari.load(
        env_name,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=[AtariPreprocessing, FrameStack4]
    )
    # env_name = "CartPole-v0"
    # env = suite_gym.load(env_name)

    train_env = tf_py_environment.TFPyEnvironment(env)
    eval_env = tf_py_environment.TFPyEnvironment(env)

    num_iterations = 20000  # @param {type:"integer"}
    replay_buffer_max_length = 1000000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    preprocessing_layer = tf.keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params = [512]

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params
    )

    train_step_counter = tf.Variable(0)
    update_period = 4
    entropy_tau = 0.9
    alpha = 0.3
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.95, momentum=0.0, epsilon=0.00001, centered=True)
    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1.0, decay_steps=250000 // update_period, end_learning_rate=0.01)

    agent = DPPAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=2000,
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        gamma=0.99,
        train_step_counter=train_step_counter,
        epsilon_greedy=lambda: epsilon_fn(train_step_counter),
        entropy_tau=entropy_tau,
        alpha=alpha
    )

    agent.initialize()

    eval_policy = agent.policy  # greedy
    collect_policy = agent.collect_policy  # epsilon greedy
    # q_policy = tf_agents.policies.q_policy.QPolicy(agent.time_step_spec, agent.action_spec, agent._q_network)
    # collect_policy = tf_agents.policies.boltzmann_policy.BoltzmannPolicy(q_policy, 0.3)

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
        if (iteration + 1) % 100 == 0:
            print("average return: ", train_metrics[1].result().numpy())
