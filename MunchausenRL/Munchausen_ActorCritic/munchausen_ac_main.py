import tempfile
from Munchausen_ActorCritic import MunchausenACAgent, ShowProgress
import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet, tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.experimental.train.utils import spec_utils

from tf_agents.experimental.train.utils import train_utils
from tf_agents.networks import actor_distribution_network
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import random_tf_policy


if __name__ == '__main__':

    tempdir = tempfile.gettempdir()
    env_name = "MinitaurBulletEnv-v0"

    writer = tf.summary.create_file_writer("MunchausenAC_logs/")

    num_iterations = 500000
    initial_collect_steps = 10000
    collect_step_per_iteration = 1
    replay_buffer_capacity = 10000

    batch_size = 64

    critic_learning_rate = 3e-4
    actor_learning_rate = 3e-4
    alpha_learning_rate = 3e-4
    target_update_tau = 0.005
    target_update_period = 1
    gamma = 0.99
    reward_scale_factor = 1.0

    actor_fc_layer_params = (256, 256)
    critic_joint_fc_layer_params = (256, 256)

    log_interval = 5000
    num_eval_episodes = 20
    eval_interval = 10000
    policy_save_interval = 5000

    collect_env = tf_py_environment.TFPyEnvironment(suite_pybullet.load(env_name))
    eval_env = tf_py_environment.TFPyEnvironment(suite_pybullet.load(env_name))

    observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(collect_env))

    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer=tf.keras.initializers.HeNormal(),
        last_kernel_initializer=tf.keras.initializers.HeNormal()
    )

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layer_params,
        continuous_projection_net=(tanh_normal_projection_network.TanhNormalProjectionNetwork)
    )


    sigma = 0.9
    train_step = train_utils.create_train_step()

    agent = MunchausenACAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        sigma=sigma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step
    )

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy



    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=collect_env.batch_size,
        max_length=replay_buffer_capacity
    )

    dataset = replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2).prefetch(3)

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    driver = dynamic_step_driver.DynamicStepDriver(
        collect_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=4
    )

    # collect some experience before training
    initial_collect_policy = random_tf_policy.RandomTFPolicy(collect_env.time_step_spec(), collect_env.action_spec())

    init_driver = dynamic_step_driver.DynamicStepDriver(
        collect_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch, ShowProgress(2000)],
        num_steps=2000,
    )

    final_time_step, final_policy_state = init_driver.run()

    agent.train = common.function(agent.train)
    driver.run = common.function(driver.run)

    time_step = None
    policy_state = agent.collect_policy.get_initial_state(collect_env.batch_size)
    iterator = iter(dataset)

    for iteration in range(num_iterations):
        time_step, policy_state = driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss: {:.5f}".format(iteration, train_loss.loss.numpy()), end="")
        if (iteration + 1) % 100 == 0:
            print("average return: ", train_metrics[1].result().numpy())
            with writer.as_default():
                tf.summary.scalar("Mean return", train_metrics[1].result().numpy(), step=iteration)
                tf.summary.scalar("loss", train_loss.loss.numpy(), step=iteration)