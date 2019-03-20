import neural_network_configuration as conf
import numpy as np
import scipy.special as ss


def sigmoid(z):
  """
  sigmoid/softmax activation, uses ss.expit to avoid numeric instability
  :param z: vectorized values to be activated
  :return: vectorized activated values
  """
  return ss.expit(z)


def cross_entropy(h, y):
  """
  cross entropy cost function
  :param h: vectorized hypothesis value
  :param y: vectorized actual value
  :return: cross entropy loss
  """
  return -np.sum(y * np.log(h+1e-6))


def activation(prev, weights, bias):
  """
  a neural network layer
  :param prev: activations of previous layer
  :param weights: weights of this layer
  :param bias: biases of this layer
  :return: activations of this layer
  """
  prev_copy = np.r_[np.ones(prev.shape[1])[np.newaxis], prev]
  weights_copy = np.c_[bias, weights]
  return sigmoid(np.matmul(weights_copy, prev_copy))


def feed_forward(x, weights, bias):
  """
  feeds forward the input through the neural network
  :param x: input
  :param weights: weights
  :param bias: biases
  :return: activations for all the layers (last layer is the output)
  """
  # activation value matrices of the two twin networks and the joined network
  activations = np.ndarray(conf.LAYERS_NUM, dtype=np.matrix)

  # transposing horizontal input vectors (or matrices) into feature vectors
  if len(x.shape) == 1:
    activations[0] = x[np.newaxis].T
  else:
    activations[0] = x.T

  # forward propagation
  for i in range(1, conf.LAYERS_NUM):
    activations[i] = activation(activations[i - 1], weights[i - 1], bias[i - 1])

  return activations


def regularize(weights, cost, gradients):
  """
  regularize the gradients and costs
  :param weights: weights
  :param cost: cost function value
  :param gradients: gradients value
  :return: regularized cost and gradietns
  """
  for n in range(1, conf.LAYERS_NUM):
    weights_without_bias = np.c_[(np.zeros(weights[n - 1].shape[0]),
                                  weights[n - 1])]
    regularization_offset = conf.REG_CONST * weights_without_bias
    gradients[n - 1] += regularization_offset
    cost += conf.REG_CONST * np.sum(np.multiply(weights[n - 1], weights[n - 1])) / 2
  return cost, gradients


def cost_derivatives(x, y, weights, bias):
  """
  gets the gradients of cost w.r.t. biases & weights
  :param x: input
  :param y: vectorized labels
  :param weights: weights
  :param bias: biases
  :return: cost and gradients(for both biases and weights)
  """
  # zero initializes cost and gradients
  cost = np.float(0)
  transformations_derivatives = np.ndarray(conf.LAYERS_NUM - 1,
                                           dtype=np.ndarray)
  gradients = np.ndarray(conf.LAYERS_NUM - 1, dtype=np.matrix)
  for i in range(1, conf.LAYERS_NUM):
    gradients[i - 1] = np.matrix(
        np.zeros((conf.LAYERS_UNITS[i], conf.LAYERS_UNITS[i - 1] + 1)))

  # sum up the derivatives of cost for each sample
  activations = feed_forward(x, weights, bias)
  for i in range(0, x.shape[0]):
    cost += cross_entropy(activations[conf.LAYERS_NUM - 1][:, i], y[:, i])

    # backpropagate
    transformations_derivatives[conf.LAYERS_NUM - 2] = \
        activations[conf.LAYERS_NUM - 1][:, i] - y[:, i]

    for n in reversed(range(0, conf.LAYERS_NUM - 2)):
      # n is the n + 1 layer in the network
      next_layer_transforms_gradients = transformations_derivatives[
          n + 1]
      next_layer_weights = weights[n + 1]
      this_layer_activations_gradients = activations[n + 1][:, i] \
                                         * (1 - activations[n + 1][:, i])
      transformations_derivatives[n] = np.multiply(
          np.matmul(next_layer_weights.T, next_layer_transforms_gradients),
          this_layer_activations_gradients)

    # calculate gradients of weights in relation to their transformations
    for n in range(1, conf.LAYERS_NUM):
      ad = np.r_[np.ones(1), activations[n - 1][:, i]][np.newaxis]
      gradients[n - 1] += \
          np.matmul(transformations_derivatives[n - 1][np.newaxis].T, ad)

  # take their mean
  cost /= x.shape[0]
  for n in range(1, conf.LAYERS_NUM):
    gradients[n - 1] /= x.shape[0]

  return cost, gradients


def predict(x, weights=None, bias=None):
  """
  gives discrete output (labels) based on input (vectorized images)
  :param x: input
  :param weights: weights of the neural network
  :param bias: biases of the neural network
  :return: predicted labels
  """
  if weights is None:
    weights = np.ndarray(conf.LAYERS_NUM - 1, dtype=np.matrix)
    bias = np.ndarray(conf.LAYERS_NUM - 1, dtype=np.ndarray)
    weights[0] = np.load("neural_network/weights_1.npy")
    weights[1] = np.load("neural_network/weights_2.npy")
    bias[0] = np.load("neural_network/bias_1.npy")
    bias[1] = np.load("neural_network/bias_2.npy")
  h = feed_forward(x, weights, bias)
  return np.argmax(h[-1], axis=0)


def accuracy(x, labels, weights, bias):
  """
  a performance metric
  :param x: dataset inputs
  :param labels: dataset labels
  :param weights: neural network weights
  :param bias: neural network biases
  :return: accuracy between 0 and 1
  """
  out = predict(x, weights, bias)
  results = [(out[i] == labels[i])
             for i in range(x.shape[0])]
  return sum(result for result in results) / x.shape[0]


# def numerical_derivative_approximation(x, y, weights, bias, i, j, l, cost):
#   # make two copies of the weights and biases
#   weights_copy = np.ndarray(nn_conf.LAYERS_NUM - 1, dtype=np.matrix)
#   bias_copy = np.ndarray(nn_conf.LAYERS_NUM - 1, dtype=np.ndarray)
#   for n in range(1, nn_conf.LAYERS_NUM):
#     weights_copy[n - 1] = weights[n - 1]
#     bias_copy[n - 1] = bias[n - 1]
#
#   # copy and modify the weight/bias matrices at (i, j, l)
#   if j == 0:
#     new_bias = np.ndarray.copy(bias_copy[l])
#     new_bias[i] += nn_conf.NUMERICAL_DELTA
#     bias_copy[l] = new_bias
#   else:
#     new_weights = np.ndarray.copy(weights_copy[l])
#     # j - 1 due to lack of biases
#     new_weights[i][j - 1] += nn_conf.NUMERICAL_DELTA
#     weights_copy = new_weights
#   # forward propagate
#   out = feed_forward(x, weights_copy, bias_copy)
#   # calculate costs for both sets of weights
#   new_cost = cross_entropy(out[nn_conf.LAYERS_NUM - 1], y)
#   # print("numerical costs: " + str(cost_1) + ", " + str(cost_2))
#   return (new_cost - cost) / nn_conf.NUMERICAL_DELTA
#
#
# def num_approx_aggregate(x, y, weights, bias):
#   out = feed_forward(x, weights, bias)
#   cost = cross_entropy(out[nn_conf.LAYERS_NUM - 1], y)
#   weights_gradients = np.ndarray(nn_conf.LAYERS_NUM - 1, dtype=np.matrix)
#   for l in range(nn_conf.LAYERS_NUM - 1):
#     mat = np.zeros(shape=(nn_conf.LAYERS_UNITS[l + 1], nn_conf.LAYERS_UNITS[l] + 1))
#     for i in range(nn_conf.LAYERS_UNITS[l + 1]):
#       for j in range(nn_conf.LAYERS_UNITS[l] + 1):
#         mat[i][j] = numerical_derivative_approximation(x, y, weights, bias,
#                                                        i, j, l, cost)
#     weights_gradients[l] = mat
#   return weights_gradients
