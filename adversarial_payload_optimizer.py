import data_preparation as dp
import numpy as np
import neural_network_configuration as nn_conf
import neural_network as nn
import adversarial_configuration as conf


def payload_derivatives(payload, desired_output, weights, bias):
  """
  gets the gradients of cost w.r.t. payload pixels
  :param payload: inputs
  :param desired_output: vectorized label
  :param weights: weights
  :param bias: biases
  :return: optimization objective (cost) and gradients
  """
  # zero initializes cost and gradients
  cost = np.float(0)
  transformations_derivatives = np.ndarray(nn_conf.LAYERS_NUM - 1,
                                           dtype=np.ndarray)

  activations = nn.feed_forward(payload, weights, bias)
  # minimizing this cost is our optimization objective
  # cross entropy gives us the distance between nn behavior and desired behavior
  cost += nn.cross_entropy(activations[nn_conf.LAYERS_NUM - 1], desired_output)
  # backpropagate
  transformations_derivatives[nn_conf.LAYERS_NUM - 2] = \
    activations[nn_conf.LAYERS_NUM - 1] - desired_output
  for n in reversed(range(0, nn_conf.LAYERS_NUM - 2)):
    # n is the n + 1 layer in the network
    next_layer_transforms_gradients = transformations_derivatives[
        n + 1]
    next_layer_weights = weights[n + 1]
    this_layer_activations_gradients = activations[n + 1] \
                                       * (1 - activations[n + 1])
    transformations_derivatives[n] = np.multiply(
        np.matmul(next_layer_weights.T, next_layer_transforms_gradients),
        this_layer_activations_gradients)
  payload_gradients = np.matmul(weights[0].T, transformations_derivatives[0])
  return cost, payload_gradients


def optimize_payload(desired_output, original, label):
  """
  optimizes a payload
  :param desired_output: attacker's desired behavior of the neural network
  :param original: the image
  :return: optmized payload
  """
  # load configuration from target neural network
  neural_network_weights = np.ndarray(nn_conf.LAYERS_NUM - 1, dtype=np.matrix)
  neural_network_bias = np.ndarray(nn_conf.LAYERS_NUM - 1, dtype=np.ndarray)
  neural_network_weights[0] = np.load("neural_network/weights_1.npy")
  neural_network_weights[1] = np.load("neural_network/weights_2.npy")
  neural_network_bias[0] = np.load("neural_network/bias_1.npy")
  neural_network_bias[1] = np.load("neural_network/bias_2.npy")
  payload = np.copy(original)
  epoch = 0
  while True:
    cost, gradients = payload_derivatives(payload, desired_output,
                                          neural_network_weights,
                                          neural_network_bias)
    difference = payload - original
    constraint = conf.LAMBDA * np.sum(np.multiply(difference, difference)) / 2
    cost += constraint
    constraint_gradients = difference
    payload -= conf.LEARNING_RATE * (np.squeeze(np.asarray(gradients)) + conf.LAMBDA * constraint_gradients)
    epoch += 1
    if epoch % conf.EVAL_INTERVAL == 0:
      print("Epoch %d, Cost %f" % (epoch, cost))
      neural_output = nn.feed_forward(payload, neural_network_weights, neural_network_bias)[-1]
      if neural_output[label] > conf.THRESHOLD and np.argmax(neural_output) == label and 0.1 > cost:
        print(neural_output[label])
        print(np.argmax(neural_output))
        print("Mission Accomplished")
        break
  return payload


def generate_payload(output_num, looklike_num):
  desired_output = dp.vectorized_label(output_num)
  original_image = dp.normalize(dp.load_sample_image(looklike_num))
  payload = optimize_payload(desired_output, original_image, output_num)
  dp.save_image(dp.denormalize(original_image * -1), "payload/original.png")
  dp.save_image(dp.denormalize(payload * -1), "payload/output.png")


def main():
  output_num = int(input("What number do you want the neural network to classify the payload as?\n"))
  looklike_num = int(input("What number do you want the payload to look like?\n"))
  generate_payload(output_num, looklike_num)


if __name__ == "__main__":
  main()
