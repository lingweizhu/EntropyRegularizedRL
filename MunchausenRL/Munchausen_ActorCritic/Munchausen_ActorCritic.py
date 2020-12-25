import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.agents import data_converter
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity
from tf_agents.agents.sac import sac_agent

import collections
from typing import Callable, Optional, Text

SacLossInfo = collections.namedtuple(
    'SacLossInfo', ('critic_loss', 'actor_loss', 'alpha_loss'))

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")

class MunchausenACAgent(sac_agent.SacAgent):

    def __init__(self,
                 time_step_spec: ts.TimeStep,
                 action_spec: types.NestedTensorSpec,
                 critic_network: network.Network,
                 actor_network: network.Network,
                 actor_optimizer: types.Optimizer,
                 critic_optimizer: types.Optimizer,
                 alpha_optimizer: types.Optimizer,
                 actor_loss_weight: types.Float = 1.0,
                 critic_loss_weight: types.Float = 0.5,
                 alpha_loss_weight: types.Float = 1.0,
                 actor_policy_ctor: Callable[
                     ..., tf_policy.TFPolicy] = actor_policy.ActorPolicy,
                 critic_network_2: Optional[network.Network] = None,
                 target_critic_network: Optional[network.Network] = None,
                 target_critic_network_2: Optional[network.Network] = None,
                 target_update_tau: types.Float = 1.0,
                 target_update_period: types.Int = 1,
                 td_errors_loss_fn: types.LossFn = tf.math.squared_difference,
                 gamma: types.Float = 1.0,
                 sigma: types.Float = 0.9,
                 reward_scale_factor: types.Float = 1.0,
                 initial_log_alpha: types.Float = 0.0,
                 use_log_alpha_in_alpha_loss: bool = True,
                 target_entropy: Optional[types.Float] = None,
                 gradient_clipping: Optional[types.Float] = None,
                 debug_summaries: bool = False,
                 summarize_grads_and_vars: bool = False,
                 train_step_counter: Optional[tf.Variable] = None,
                 name: Optional[Text] = None):

        tf.Module.__init__(self, name=name)

        self._check_action_spec(action_spec)

        net_observation_spec = time_step_spec.observation
        critic_spec = (net_observation_spec, action_spec)

        self._critic_network_1 = critic_network

        if critic_network_2 is not None:
            self._critic_network_2 = critic_network_2
        else:
            self._critic_network_2 = critic_network.copy(name='CriticNetwork2')
            # Do not use target_critic_network_2 if critic_network_2 is None.
            target_critic_network_2 = None

        # Wait until critic_network_2 has been copied from critic_network_1 before
        # creating variables on both.
        self._critic_network_1.create_variables(critic_spec)
        self._critic_network_2.create_variables(critic_spec)

        if target_critic_network:
            target_critic_network.create_variables(critic_spec)

        self._target_critic_network_1 = (
            common.maybe_copy_target_network_with_checks(
                self._critic_network_1,
                target_critic_network,
                input_spec=critic_spec,
                name='TargetCriticNetwork1'))

        if target_critic_network_2:
            target_critic_network_2.create_variables(critic_spec)
        self._target_critic_network_2 = (
            common.maybe_copy_target_network_with_checks(
                self._critic_network_2,
                target_critic_network_2,
                input_spec=critic_spec,
                name='TargetCriticNetwork2'))

        if actor_network:
            actor_network.create_variables(net_observation_spec)
        self._actor_network = actor_network

        policy = actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=False)

        self._train_policy = actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=True)

        self._log_alpha = common.create_variable(
            'initial_log_alpha',
            initial_value=initial_log_alpha,
            dtype=tf.float32,
            trainable=True)

        if target_entropy is None:
            target_entropy = self._get_default_target_entropy(action_spec)

        self._use_log_alpha_in_alpha_loss = use_log_alpha_in_alpha_loss
        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._alpha_optimizer = alpha_optimizer
        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._alpha_loss_weight = alpha_loss_weight
        self._td_errors_loss_fn = td_errors_loss_fn
        self._gamma = gamma
        self._reward_scale_factor = reward_scale_factor
        self._target_entropy = target_entropy
        self._gradient_clipping = gradient_clipping
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._update_target = self._get_target_updater(
            tau=self._target_update_tau, period=self._target_update_period)

        self.sigma = sigma

        train_sequence_length = 2 if not critic_network.state_spec else None

        super(sac_agent.SacAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy=policy,
            collect_policy=policy,
            train_sequence_length=train_sequence_length,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
            validate_args=False
        )

        self._as_transition = data_converter.AsTransition(
            self.data_context, squeeze_time_dim=(train_sequence_length == 2))


    def critic_loss(self,
                    time_steps: ts.TimeStep,
                    actions: types.Tensor,
                    next_time_steps: ts.TimeStep,
                    td_errors_loss_fn: types.LossFn,
                    gamma: types.Float = 1.0,
                    reward_scale_factor: types.Float = 1.0,
                    weights: Optional[types.Tensor] = None,
                    training: bool = False) -> types.Tensor:
        """Computes the critic loss for SAC training.
        Args:
          time_steps: A batch of timesteps.
          actions: A batch of actions.
          next_time_steps: A batch of next timesteps.
          td_errors_loss_fn: A function(td_targets, predictions) to compute
            elementwise (per-batch-entry) loss.
          gamma: Discount for future rewards.
          reward_scale_factor: Multiplicative factor to scale rewards.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
          training: Whether this loss is being used for training.
        Returns:
          critic_loss: A scalar critic loss.
        """
        with tf.name_scope('critic_loss'):
            nest_utils.assert_same_structure(actions, self.action_spec)
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)
            nest_utils.assert_same_structure(next_time_steps, self.time_step_spec)

            next_actions, next_log_pis = self._actions_and_log_probs(next_time_steps)
            curr_actions, curr_log_pis = self._actions_and_log_probs(time_steps)
            target_input = (next_time_steps.observation, next_actions)
            target_q_values1, unused_network_state1 = self._target_critic_network_1(
                target_input, next_time_steps.step_type, training=False)
            target_q_values2, unused_network_state2 = self._target_critic_network_2(
                target_input, next_time_steps.step_type, training=False)
            target_q_values = (
                    tf.minimum(target_q_values1, target_q_values2) -
                    tf.exp(self._log_alpha) * next_log_pis)

            td_targets = tf.stop_gradient(
                reward_scale_factor * next_time_steps.reward +
                gamma * next_time_steps.discount * target_q_values +
                self.sigma * tf.exp(self._log_alpha) * curr_log_pis  # Munchausen term
            )

            pred_input = (time_steps.observation, actions)
            pred_td_targets1, _ = self._critic_network_1(
                pred_input, time_steps.step_type, training=training)
            pred_td_targets2, _ = self._critic_network_2(
                pred_input, time_steps.step_type, training=training)
            critic_loss1 = td_errors_loss_fn(td_targets, pred_td_targets1)
            critic_loss2 = td_errors_loss_fn(td_targets, pred_td_targets2)
            critic_loss = critic_loss1 + critic_loss2

            if critic_loss.shape.rank > 1:
                # Sum over the time dimension.
                critic_loss = tf.reduce_sum(
                    critic_loss, axis=range(1, critic_loss.shape.rank))

            agg_loss = common.aggregate_losses(
                per_example_loss=critic_loss,
                sample_weight=weights,
                regularization_loss=(self._critic_network_1.losses +
                                     self._critic_network_2.losses))
            critic_loss = agg_loss.total_loss

            self._critic_loss_debug_summaries(td_targets, pred_td_targets1,
                                              pred_td_targets2)

            return critic_loss


    def actor_loss(self,
                   time_steps: ts.TimeStep,
                   weights: Optional[types.Tensor] = None) -> types.Tensor:
        """Computes the actor_loss for SAC training.
        Args:
          time_steps: A batch of timesteps.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
        Returns:
          actor_loss: A scalar actor loss.
        """
        with tf.name_scope('actor_loss'):
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)

            actions, log_pi = self._actions_and_log_probs(time_steps)
            target_input = (time_steps.observation, actions)
            target_q_values1, _ = self._critic_network_1(
                target_input, time_steps.step_type, training=False)
            target_q_values2, _ = self._critic_network_2(
                target_input, time_steps.step_type, training=False)
            target_q_values = tf.minimum(target_q_values1, target_q_values2)
            # currently policy loss still follows standard SAC
            actor_loss = tf.exp(self._log_alpha) * log_pi - target_q_values
            if actor_loss.shape.rank > 1:
                # Sum over the time dimension.
                actor_loss = tf.reduce_sum(
                    actor_loss, axis=range(1, actor_loss.shape.rank))
            reg_loss = self._actor_network.losses if self._actor_network else None
            agg_loss = common.aggregate_losses(
                per_example_loss=actor_loss,
                sample_weight=weights,
                regularization_loss=reg_loss)
            actor_loss = agg_loss.total_loss
            self._actor_loss_debug_summaries(actor_loss, actions, log_pi,
                                             target_q_values, time_steps)

            return actor_loss