# Two-Hidden-Layer-MLP-with-Explicit-Backpropagation-for-MNIST-classification

In this project, our goal is to implement techniques to learn a deep neural network with two hidden layers and write code for backpropagation from scratch without
using library functions for performing image digit classification. The MNIST
dataset has been used as the training and testing dataset. The neural network has been trained by
minimizing the Negative Log Likelihood or cross-entropy loss function of the output nodes.
We tune our parameters by minimizing the loss function over several epochs by gradient
descent approach. We have implemented Stochastic Gradient Descent technique. We have
also applied L1 norm regularization of parameters to prevent the model from overfitting. In
order to apply gradient descent, we apply backpropagation method, which we have calculated explicitly, without using library functions. the gradient of
the output is first computed and the gradients pass to the previous layers and finally we
update the parameters after each iteration
