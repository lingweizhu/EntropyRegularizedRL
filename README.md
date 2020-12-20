# Entropy-regularized Reinforcement Learning

This repository contains my attempt to build entropy-regularized reinforcement learning algorithms. For convenience I just subclass existing modules from [TF Agents](https://github.com/tensorflow/agents) for replay buffer and driver. Those work efficiently under tensorflow framework. Make sure you have tensorflow and TF Agents installed.

Might extend to a complete version later.


## Dynamic Policy Programming and Conservative Value Iteration
I used the same architecture for all algorithms here. You can modify the algorithm to be either on-policy or off-policy by changing the collect policy which I commented out in the main file (as DQN is off-policy and DPP is on-policy).

Check the foundational paper [Dynamic Policy Programming](https://jmlr.org/papers/volume13/azar12a/azar12a.pdf) (DPP) for the theoretical part. [Conservative Value Iteration](http://proceedings.mlr.press/v89/kozuno19a/kozuno19a.pdf) provides more theoretical insights on the roles played by Softmax/KL/Action Gap. 

There has been many extensions of DPP on practical problems, see e.g. [Deep Dual DPP for robot control](https://ieeexplore.ieee.org/document/8205960), [Kernel DPP for robot control](https://www.sciencedirect.com/science/article/pii/S0893608017301430) and [Factorial Fastfood DPP for plant control](https://www.sciencedirect.com/science/article/pii/S0967066120300186).

Note that by its nature DPP tends to diverge (**Corollary 4**. of DPP paper), check also the paper **Momemtum in Reinforcement Learning** (below) for discussion. If you want to implement DPP or CVI for very long horizon tasks, you might want to clip the preference function P or estimate the Q function instead and add log-policy to it, instead of directly estimating P.


## Momentum in Reinforcement Learning
Adding KL divergence in the value function formulation allows us to obtain analytic maximizer as soft of Boltzmann distribution. By induction the policy is the average of all previous Q functions. Check this NeuralIPS 2020 spotlight paper [Leverage the average](https://arxiv.org/abs/2003.14089) for details. The average of Q functions serves like momentum in optimization literature. 

Hence we might directly leverage this idea rather than adding KL divergence. This is the idea of [Momentum in Reinforcement Learning](https://arxiv.org/abs/1910.09322). Note that in Momentum-DQN the collect policy is with respect to the *H*-network.


## Munchausen Reinforcement Learning
While all the above show the merits of KL divergence regularization, they are **explicit**, hence the great linear loss bound (Thm 1 of Leverage the average) does not hold for deep implementation. However, it holds when we do KL regularization **implicitly**. Check the paper [Munchausen Reinforcement Learning](https://arxiv.org/abs/2007.14430) for details. This is achieved by simply adding a log-policy term to the value function. 

Extending it to actor-critic architecture is under exploration. 
