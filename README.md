# MNIST Neural Network Adversarial Attacks
A demonstration of gradient-based adversarial attacks on a simple 90% accuracy implementation of 
a MNIST hand-written digit classification neural network with one hidden layer and 
32 hidden units. The attacker differentiates the neural network to make it misclassify an image that is euclideanly close to a handwritten "i" as a "j", where 0 <= i <= 9, 0 <= j <= 9, and i =\= j.
 
Usage: python3 adversarial_payload_optimizer.py

Required libraries: numpy, matplotlib

Adjust optimization parameters in adversarial_configuration.py (lambda, threshold, and learning rate primarily)
as you see fit.
