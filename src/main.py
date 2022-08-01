from cgi import test
import random
from NeuralNetworkLib.ErrorFunctions.CrossEntropyWithSoftMax import CrossEntropyWithSoftMax
from NeuralNetworkLib.Network import Network
from NeuralNetworkLib.ActivationFunctions.ReLU import ReLU
from NeuralNetworkLib.ActivationFunctions.Sigmoid import Sigmoid
from NeuralNetworkLib.ActivationFunctions.Identity import Identity
from NeuralNetworkLib.DataLoader import DataLoader
from NeuralNetworkLib.Layers.FullyConnectedLayer import FullyConnectedLayer
from NeuralNetworkLib.ErrorFunctions.CrossEntropy import CrossEntropy
import os
import matplotlib.pyplot as plt
import numpy as np

dataset_path = os.path.normpath(os.path.join(os.getcwd(), "../dataset/mnist/"))
data_loader = DataLoader(dataset_path, dataset_percentage=0.1, training_set_percentage=0.75) #loads mnist, splitting it into 75% training and 25% validation
data_loader.LoadDataset()

net = Network(data_loader, CrossEntropyWithSoftMax, learning_rate=0.4)

random.seed(3)

number_of_hidden_layers = 2
number_of_nodes = 5
number_of_output_nodes = 10
input_size = len(data_loader.train_X[0])

input_layer = FullyConnectedLayer(input_size, input_size, activation_function=Identity)
net.add_layer(input_layer)

input_layer = FullyConnectedLayer(input_size, number_of_nodes, activation_function=Sigmoid)
net.add_layer(input_layer)

for i in range(0, number_of_hidden_layers):
    layer = FullyConnectedLayer(number_of_nodes, number_of_nodes, activation_function=Sigmoid)
    net.add_layer(layer)

output_layer = FullyConnectedLayer(number_of_nodes, number_of_output_nodes, activation_function=Identity)
net.add_layer(output_layer)

net.train(MAX_EPOCH=10)

test_accuracy = net.compute_test_accuracy()
print(f"Test accuracy: {test_accuracy}")

print(net.get_class(data_loader.train_X[10]))
image = np.reshape(data_loader.train_X[10], (28,28)) # 28 = sqrt(sample size)
fig = plt.figure
plt.imshow(image, cmap='gray')
plt.show()



