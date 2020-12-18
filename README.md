# Munchausen Reinforcement Learning

This is an implementation of [Munchausen Reinforcement Learning](https://arxiv.org/abs/2007.14430) based on [TF Agents](https://github.com/tensorflow/agents).

Currently it is only subclassing DQN module from TF Agents and rewrite the loss function. Might extend to a complete version later.

## Relationship
DPP-DQN uses the same architecture for implementing [Dynamic Policy Programming](https://jmlr.org/papers/volume13/azar12a/azar12a.pdf) (DPP). Check [Conservative Value Iteration](http://proceedings.mlr.press/v89/kozuno19a/kozuno19a.pdf) which is a generalization of DPP also.

Note that by its nature DPP tends to diverge (**Corollary 4**. of DPP paper), check also [this recent paper](https://arxiv.org/abs/1910.09322) for discussion. If you want to implement DPP or CVI for very long horizon tasks, you might want to clip the value function $\Psi$ or estimate Q function and add a $\log\pi(a|s)$ term with it, instead of directly estimating $\Psi$.
