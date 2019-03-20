import neural_network_configuration as conf
import numpy as np
import datasets_loader.mnist
import PIL.Image
import scipy.misc


def vectorized_label(label):
  """
  turns a label into a vector for training
  :param label: scalar
  :return: vector
  """
  y = np.zeros((10, 1))
  y[label] = 1.0
  return y


def generate_x_y():
  """
  generates FAKE_SAMPLE_SIZE number of rows of fake images
  :return: input, vectorized labels, scalar labels
  """
  x_dim = (conf.FAKE_SAMPLE_SIZE, conf.IMAGE_SIZE)
  y_dim = (10, conf.FAKE_SAMPLE_SIZE)
  x = np.random.rand(x_dim[0], x_dim[1])
  labels =  np.random.choice(a=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=y_dim[1], p=[0.2, 0, 0, 0, 0.3, 0, 0, 0, 0, 0.5])
  y = np.hstack([vectorized_label(label) for label in labels])
  return x, y, labels


def normalize(x):
  """
  normalizes the input
  :param x: input
  :return: normalized input
  """
  return (x - 255 / 2) / 255


def denormalize(x):
  """
  un-normalizes the input
  :param x: input
  :return: un-normalized input
  """
  return x * 255 + 255 / 2


def load_datasets():
  """
  loads and normalizes the mnist dataset
  :return: training and testing datasets
  """
  x_train, l_train, x_test, l_test = datasets_loader.mnist.load()
  y_train = np.hstack([vectorized_label(label) for label in l_train])
  y_test = np.hstack([vectorized_label(label) for label in l_test])
  return normalize(x_train), y_train, normalize(x_test), y_test, l_train, l_test


def save_image(x, filename, normalized=False):
  """
  turns one row of gray scale values into a png image, then saves it
  :param x: vector
  """
  if normalized:
    x = denormalize(x)
  matrix = np.reshape(x, (conf.IMAGE_LENGTH, conf.IMAGE_LENGTH))
  img = PIL.Image.fromarray(matrix, "L")
  img.save(filename)


def find_and_save_sample_images():
  x_train, l_train, _, _ = datasets_loader.mnist.load()
  images = x_train
  labels = l_train
  needed = [True] * 10
  found = 0
  for i in range(labels.shape[0]):
    label = labels[i]
    if needed[label]:
      save_image(images[i], "sample_images/" + str(label) + ".png")
      needed[label] = False
      found += 1
    if found == 10:
      break

def load_sample_image(label):
  filepath = "sample_images/" + str(label) + ".png"
  normalized_vector = normalize(scipy.misc.imread(filepath).flatten())
  return normalized_vector


def load_image(filepath):
  normalized_vector = normalize(scipy.misc.imread(filepath).flatten())
  return normalized_vector


# if __name__ == "__main__":
#   find_and_save_sample_images()
