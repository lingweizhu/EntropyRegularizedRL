from tf_agents.agents.dqn import dqn_agent

import tensorflow as tf

from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils
import collections
from typing import Optional, Text


class DqnLossInfo(collections.namedtuple('DqnLossInfo',
                                         ('td_loss', 'td_error'))):
    pass


def compute_dpp_td_targets(next_p_values, p_target_values,
                                  actions, rewards, multi_dim_actions,
                                  discounts, alpha, entropy_tau):

    boltzmann_p = tf.reduce_sum(tf.nn.softmax(p_target_values / entropy_tau, axis=1) * p_target_values, 1)

    p_target_values = common.index_with_actions(p_target_values,
                              tf.cast(actions, dtype=tf.int32),
                              multi_dim_actions)

    action_gap = alpha * (p_target_values - boltzmann_p)

    next_boltzmann_p = tf.reduce_sum(tf.nn.softmax(next_p_values / entropy_tau, axis=1) * next_p_values, 1)

    td_targets = rewards + discounts * next_boltzmann_p + action_gap

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




class DPPAgent(dqn_agent.DqnAgent):

    def __init__(
            self,
            time_step_spec: ts.TimeStep,
            action_spec: types.NestedTensorSpec,
            q_network: network.Network,
            optimizer: types.Optimizer,
            observation_and_action_constraint_splitter: Optional[
                types.Splitter] = None,
            epsilon_greedy: types.Float = 0.1,
            n_step_update: int = 1,
            boltzmann_temperature: Optional[types.Int] = None,
            emit_log_probability: bool = False,
            # Params for target network updates
            target_q_network: Optional[network.Network] = None,
            target_update_tau: types.Float = 1.0,
            target_update_period: int = 1,
            # Params for training.
            td_errors_loss_fn: Optional[types.LossFn] = None,
            gamma: types.Float = 1.0,
            reward_scale_factor: types.Float = 1.0,
            gradient_clipping: Optional[types.Float] = None,
            # Params for debugging
            debug_summaries: bool = False,
            summarize_grads_and_vars: bool = False,
            train_step_counter: Optional[tf.Variable] = None,
            name: Optional[Text] = None,
            entropy_tau: types.Float = 0.1,
            alpha: types.Float = 0.95
    ):

        tf.Module.__init__(self, name=name)

        self._check_action_spec(action_spec)

        if epsilon_greedy is not None and boltzmann_temperature is not None:
            raise ValueError(
                'Configured both epsilon_greedy value {} and temperature {}, '
                'however only one of them can be used for exploration.'.format(
                    epsilon_greedy, boltzmann_temperature))

        self._observation_and_action_constraint_splitter = (
            observation_and_action_constraint_splitter)
        self._q_network = q_network
        net_observation_spec = time_step_spec.observation
        if observation_and_action_constraint_splitter:
            net_observation_spec, _ = observation_and_action_constraint_splitter(
                net_observation_spec)
        q_network.create_variables(net_observation_spec)
        if target_q_network:
            target_q_network.create_variables(net_observation_spec)
        self._target_q_network = common.maybe_copy_target_network_with_checks(
            self._q_network, target_q_network, input_spec=net_observation_spec,
            name='TargetQNetwork')

        self._check_network_output(self._q_network, 'q_network')
        self._check_network_output(self._target_q_network, 'target_q_network')

        self._epsilon_greedy = epsilon_greedy
        self._n_step_update = n_step_update
        self._boltzmann_temperature = boltzmann_temperature
        self._optimizer = optimizer
        self._td_errors_loss_fn = (
                td_errors_loss_fn or common.element_wise_huber_loss)
        self._gamma = gamma
        self._reward_scale_factor = reward_scale_factor
        self._gradient_clipping = gradient_clipping
        self._update_target = self._get_target_updater(
            target_update_tau, target_update_period)
        self.entropy_tau = entropy_tau
        self.alpha = alpha

        policy, collect_policy = self._setup_policy(time_step_spec, action_spec,
                                                    boltzmann_temperature,
                                                    emit_log_probability)

        if q_network.state_spec and n_step_update != 1:
            raise NotImplementedError(
                'DqnAgent does not currently support n-step updates with stateful '
                'networks (i.e., RNNs), but n_step_update = {}'.format(n_step_update))

        train_sequence_length = (
            n_step_update + 1 if not q_network.state_spec else None)

        super(dqn_agent.DqnAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=train_sequence_length,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
            validate_args=False,
        )

        if q_network.state_spec:
            # AsNStepTransition does not support emitting [B, T, ...] tensors,
            # which we need for DQN-RNN.
            self._as_transition = data_converter.AsTransition(
                self.data_context, squeeze_time_dim=False)
        else:
            # This reduces the n-step return and removes the extra time dimension,
            # allowing the rest of the computations to be independent of the
            # n-step parameter.
            self._as_transition = data_converter.AsNStepTransition(
                self.data_context, gamma=gamma, n=n_step_update)


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

            td_targets = compute_dpp_td_targets(next_p_values=next_q_values,
                                                p_target_values=q_target_values,
                                                actions=actions,
                                                rewards=reward_scale_factor * next_time_steps.reward,
                                                multi_dim_actions=self._action_spec.shape.rank > 0,
                                                discounts=gamma * next_time_steps.discount,
                                                alpha=self.alpha,
                                                entropy_tau=self.entropy_tau)

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
