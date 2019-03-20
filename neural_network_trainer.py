import neural_network_configuration as conf
import data_preparation as dp
import neural_network as nn
import numpy as np


def rand_init_weights(weights, bias):
  for i in range(1, conf.LAYERS_NUM):
    dim = (conf.LAYERS_UNITS[i], conf.LAYERS_UNITS[i - 1])
    weights[i - 1] = 10 * (np.random.rand(dim[0], dim[1]) - 0.5)
    bias[i - 1] = 10 * (np.random.rand(dim[0]) - 0.5)


def gradient_descent_fakedata():
  (x_train, y_train, l_train) = dp.generate_x_y()
  weights = np.ndarray(conf.LAYERS_NUM - 1, dtype=np.matrix)
  bias = np.ndarray(conf.LAYERS_NUM - 1, dtype=np.ndarray)
  rand_init_weights(weights, bias)
  for n in range(conf.NUM_EPOCHS):
    cost, gradients = nn.cost_derivatives(x_train, y_train, weights, bias)
    cost, gradients = nn.regularize(weights, cost, gradients)
    print("Epoch %d, Cost %f" % (n, cost))
    for l in range(conf.LAYERS_NUM - 1):
      weights[l] -= conf.LEARNING_RATE * gradients[l][:, 1:]
      bias[l] -= conf.LEARNING_RATE * np.squeeze(np.asarray(gradients[l][:, 0]))

  # cost from the last epoch
  cost = np.float(0)
  activations = nn.feed_forward(x_train, weights, bias)
  for i in range(0, x_train.shape[0]):
    cost += nn.cross_entropy(activations[conf.LAYERS_NUM - 1][:, i], y_train[:, i])
  cost /= x_train.shape[0]
  for n in range(1, conf.LAYERS_NUM):
    cost += conf.REG_CONST * np.sum(np.multiply(weights[n - 1], weights[n - 1])) / 2
  print("Epoch %d, Cost %f" % (conf.NUM_EPOCHS, cost))
  train_accuracy = nn.accuracy(x_train, l_train, weights, bias)
  print("Accuracy: %f" % train_accuracy)


def gradient_descent(weights, bias):
  x_train, y_train, x_test, y_test, l_train, l_test = dp.load_datasets()
  for n in range(conf.NUM_EPOCHS):
    cost, gradients = nn.cost_derivatives(x_train, y_train, weights, bias)
    cost, gradients = nn.regularize(weights, cost, gradients)
    if n % conf.EVAL_INTERVAL == 0:
      print("Epoch %d, Cost %f" % (n, cost))
      train_accuracy = nn.accuracy(x_train, l_train, weights, bias)
      test_accuracy = nn.accuracy(x_test, l_test, weights, bias)
      print("Train Accuracy: %f" % train_accuracy)
      print("Test Accuracy: %f" % test_accuracy)
    for l in range(conf.LAYERS_NUM - 1):
      weights[l] -= conf.LEARNING_RATE * gradients[l][:, 1:]
      bias[l] -= conf.LEARNING_RATE * np.squeeze(np.asarray(gradients[l][:, 0]))

  # cost from the last epoch
  cost = np.float(0)
  activations = nn.feed_forward(x_train, weights, bias)
  for i in range(0, x_train.shape[0]):
    cost += nn.cross_entropy(activations[conf.LAYERS_NUM - 1][:, i], y_train[:, i])
  cost /= x_train.shape[0]
  for n in range(1, conf.LAYERS_NUM):
    cost += conf.REG_CONST * np.sum(np.multiply(weights[n - 1], weights[n - 1])) / 2
  print("Epoch %d, Cost %f" % (conf.NUM_EPOCHS, cost))
  train_accuracy = nn.accuracy(x_train, l_train, weights, bias)
  test_accuracy = nn.accuracy(x_test, l_test, weights, bias)
  print("Train Accuracy: %f" % train_accuracy)
  print("Test Accuracy: %f" % test_accuracy)

  return weights, bias


def main():
  weights = np.ndarray(conf.LAYERS_NUM - 1, dtype=np.matrix)
  bias = np.ndarray(conf.LAYERS_NUM - 1, dtype=np.ndarray)
  inp = input("Continue training from saved parameters? (y/n)\n").strip()
  if inp == "y":
    weights[0] = np.load("neural_network/weights_1.npy")
    weights[1] = np.load("neural_network/weights_2.npy")
    bias[0] = np.load("neural_network/bias_1.npy")
    bias[1] = np.load("neural_network/bias_2.npy")
  else:
    rand_init_weights(weights, bias)
  weights, bias = gradient_descent(weights, bias)
  inp = input("Save trained parameters? (y/n)\n").strip()
  if inp == "y":
    np.save("neural_network/weights_1.npy", weights[0])
    np.save("neural_network/bias_1.npy", bias[0])
    np.save("neural_network/weights_2.npy", weights[1])
    np.save("neural_network/bias_2.npy", bias[1])


if __name__ == "__main__":
  main()
  #gradient_descent_fakedata()
