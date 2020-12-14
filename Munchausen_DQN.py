import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tf_agents
import collections

from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.networks import q_network
from tf_agents.policies import boltzmann_policy, random_tf_policy
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tf_agents.eval import metric_utils
from tf_agents.agents import tf_agent
from tf_agents.utils import nest_utils
# Press the green button in the gutter to run the script.

#from Munchausen_td import *

class DqnLossInfo(collections.namedtuple('DqnLossInfo',
                                         ('td_loss', 'td_error'))):
    pass


def compute_munchausen_td_targets(next_q_values, q_target_values,
                                  actions, rewards, multi_dim_actions,
                                  discounts, alpha, entropy_tau):
    tile_constant = tf.constant([1, 2], dtype=tf.int32)

    next_max_v_values = tf.expand_dims(tf.reduce_max(next_q_values, 1), -1)
    tau_logsum_next = entropy_tau * tf.reduce_logsumexp((next_q_values - next_max_v_values) / entropy_tau, axis=1)
    # batch x actions
    tau_logsum_next = tf.expand_dims(tau_logsum_next, -1)
    tau_logpi_next = next_q_values - tf.tile(next_max_v_values, tile_constant) - tf.tile(tau_logsum_next, tile_constant)

    pi_target = tf.nn.softmax(next_q_values / entropy_tau, 1)
    # valid_mask shape: (batch_size, )
    q_target = discounts * tf.reduce_sum((pi_target * (next_q_values - tau_logpi_next)), 1)  # * valid_mask

    v_target_max = tf.expand_dims(tf.reduce_max(q_target_values, 1), -1)
    tau_logsum_target = entropy_tau * tf.reduce_logsumexp((q_target_values - v_target_max) / entropy_tau, 1)
    tau_logsum_target = tf.expand_dims(tau_logsum_target, -1)
    tau_logpi_target = q_target_values - tf.tile(v_target_max, tile_constant) - tf.tile(tau_logsum_target,
                                                                                        tile_constant)

    #multi_dim_actions = self._action_spec.shape.rank > 0
    # munchausen addon uses the current state and actions
    munchausen_addon = common.index_with_actions(tau_logpi_target,
                                                 tf.cast(actions, dtype=tf.int32),
                                                 multi_dim_actions)
    #rewards = reward_scale_factor * next_time_steps.reward
    munchausen_reward = rewards + alpha * tf.clip_by_value(munchausen_addon,
                                                           clip_value_max=0,
                                                           clip_value_min=-1)
    td_targets = munchausen_reward + q_target

    return tf.stop_gradient(td_targets)


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")




class MdqnAgent(dqn_agent.DqnAgent):

    def _compute_all_q_values(self, time_steps, actions, training=False):
        network_observation = time_steps.observation

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation)

        q_values, _ = self._q_network(network_observation,
                                      step_type=time_steps.step_type,
                                      training=training)
        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        '''
        multi_dim_actions = self._action_spec.shape.rank > 0
        return common.index_with_actions(
            q_values,
            tf.cast(actions, dtype=tf.int32),
            multi_dim_actions=multi_dim_actions)
        '''
        return q_values


    def _compute_next_all_q_values(self, next_time_steps, info):
        """Compute the q value of the next state for TD error computation.
        Args:
          next_time_steps: A batch of next timesteps
          info: PolicyStep.info that may be used by other agents inherited from
            dqn_agent.
        Returns:
          A tensor of Q values for the given next state.
        """
        network_observation = next_time_steps.observation

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation)

        next_target_q_values, _ = self._target_q_network(
            network_observation, step_type=next_time_steps.step_type)
        #batch_size = (
        #        next_target_q_values.shape[0] or tf.shape(next_target_q_values)[0])
        #dummy_state = self._target_greedy_policy.get_initial_state(batch_size)
        # Find the greedy actions using our target greedy policy. This ensures that
        # action constraints are respected and helps centralize the greedy logic.
        #greedy_actions = self._target_greedy_policy.action(
        #    next_time_steps, dummy_state).action

        return next_target_q_values

    def _loss(self, experience, td_errors_loss_fn=common.element_wise_huber_loss, gamma=1.0, reward_scale_factor=1.0,
              weights=None, training=False):
        alpha = tf.constant(0.9, tf.float32)
        entropy_tau = tf.constant(0.3, tf.float32)

        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action

        valid_mask = tf.cast(~time_steps.is_last(), tf.float32)

        with tf.name_scope('loss'):
            # q_values is already gathered by actions
            q_values = self._compute_q_values(time_steps, actions, training=training)

            next_q_values = self._compute_next_all_q_values(
                next_time_steps, policy_steps.info)

            q_target_values = self._compute_next_all_q_values(time_steps, policy_steps.info)

            # This applies to any value of n_step_update and also in the RNN-DQN case.
            # In the RNN-DQN case, inputs and outputs contain a time dimension.
            #td_targets = compute_td_targets(
            #    next_q_values,
            #    rewards=reward_scale_factor * next_time_steps.reward,
            #    discounts=gamma * next_time_steps.discount)

            td_targets = compute_munchausen_td_targets(
                next_q_values=next_q_values,
                q_target_values=q_target_values,
                actions=actions,
                rewards=reward_scale_factor * next_time_steps.reward,
                discounts=gamma * next_time_steps.discount,
                multi_dim_actions=self._action_spec.shape.rank > 0,
                alpha=alpha,
                entropy_tau=entropy_tau
            )

            td_error = valid_mask * (td_targets - q_values)

            td_loss = valid_mask * td_errors_loss_fn(td_targets, q_values)

            if nest_utils.is_batched_nested_tensors(
                    time_steps, self.time_step_spec, num_outer_dims=2):
                # Do a sum over the time dimension.
                td_loss = tf.reduce_sum(input_tensor=td_loss, axis=1)

            # Aggregate across the elements of the batch and add regularization loss.
            # Note: We use an element wise loss above to ensure each element is always
            #   weighted by 1/N where N is the batch size, even when some of the
            #   weights are zero due to boundary transitions. Weighting by 1/K where K
            #   is the actual number of non-zero weight would artificially increase
            #   their contribution in the loss. Think about what would happen as
            #   the number of boundary samples increases.

            agg_loss = common.aggregate_losses(
                per_example_loss=td_loss,
                sample_weight=weights,
                regularization_loss=self._q_network.losses)
            total_loss = agg_loss.total_loss

            losses_dict = {'td_loss': agg_loss.weighted,
                           'reg_loss': agg_loss.regularization,
                           'total_loss': total_loss}

            common.summarize_scalar_dict(losses_dict,
                                         step=self.train_step_counter,
                                         name_scope='Losses/')

            if self._summarize_grads_and_vars:
                with tf.name_scope('Variables/'):
                    for var in self._q_network.trainable_weights:
                        tf.compat.v2.summary.histogram(
                            name=var.name.replace(':', '_'),
                            data=var,
                            step=self.train_step_counter)

            if self._debug_summaries:
                diff_q_values = q_values - next_q_values
                common.generate_tensor_summaries('td_error', td_error,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('td_loss', td_loss,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('q_values', q_values,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('next_q_values', next_q_values,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('diff_q_values', diff_q_values,
                                                 self.train_step_counter)

            return tf_agent.LossInfo(total_loss, DqnLossInfo(td_loss=td_loss,
                                                             td_error=td_error))
