# stochastic-ML-optimizers
This project asks the question - what if we construct an optimization algorithm based on anomalous diffusion processes?.

The SGD algorithm used for optimizing the parameters of a neural network is commonly compared to a physical particle diffusing towards the minima of some random energy landscape defined by the dataset.
This viewing the optimization problem under a physical lense yielded the widely used Stochastic Gradient Langevin Dynamics (SGLD) optimization algorithm.
Now considering the fact that not all stochasticly diffusing particles follow the Langevin Dynamics, one might ask "what if not all datasets follow the basic concept of gradient descent?".

Therefor, in this repository we implemented several tweaks of the basic SGD to create an optimization process similar to an anomalous diffusing particle
