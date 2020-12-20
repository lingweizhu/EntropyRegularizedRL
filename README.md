# Entropy-regularized Reinforcement Learning

This repository contains my attempt to build entropy-regularized reinforcement learning algorithms. For convenience I just subclass existing modules from [TF Agents](https://github.com/tensorflow/agents) for replay buffer and driver. Those work efficiently under tensorflow framework. Might extend to a complete version later.

## Munchausen Reinforcement Learning

Check the paper [Munchausen Reinforcement Learning](https://arxiv.org/abs/2007.14430) for details. The idea of **implicitly regularizing** policy iteration scheme with KL divergence is interesting. We might think of it as a reparametrization of Dynamic Policy Programming (introduced below).


## Dynamic Policy Programming and Conservative Value Iteration
DPP-DQN uses the same architecture for implementing [Dynamic Policy Programming](https://jmlr.org/papers/volume13/azar12a/azar12a.pdf) (DPP). Check [Conservative Value Iteration](http://proceedings.mlr.press/v89/kozuno19a/kozuno19a.pdf) which is a generalization of DPP also.

Note that by its nature DPP tends to diverge (**Corollary 4**. of DPP paper), check also [this recent paper](https://arxiv.org/abs/1910.09322) for discussion. If you want to implement DPP or CVI for very long horizon tasks, you might want to clip the preference function P or estimate the Q function instead and add log-policy to it, instead of directly estimating P.