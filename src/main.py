from cgi import test
import random

from NeuralNetworkLib.ErrorFunctions.CrossEntropyWithSoftMax import CrossEntropyWithSoftMax
from NeuralNetworkLib.Layers.IdentityLayer import IdentityLayer
from NeuralNetworkLib.Network import Network
from NeuralNetworkLib.ActivationFunctions.ReLU import ReLU
from NeuralNetworkLib.ActivationFunctions.Sigmoid import Sigmoid
from NeuralNetworkLib.ActivationFunctions.Identity import Identity
from NeuralNetworkLib.DataLoader import DataLoader
from NeuralNetworkLib.Layers.FullyConnectedLayer import FullyConnectedLayer
from NeuralNetworkLib.ErrorFunctions.CrossEntropy import CrossEntropy
from DummyLoader import DummyLoader
import os
import matplotlib.pyplot as plt
import numpy as np


"""
train_X = [9]
train_labels = [9]

validation_X = [5]
validation_labels = [5]

test_X = [1]
test_labels = [1]

labels = [0,1,2,3,4,5,6,7,8,9]

data_loader = DummyLoader(train_X, train_labels, validation_X, validation_labels, test_X, test_labels, labels)

net = Network(data_loader, CrossEntropyWithSoftMax, learning_rate=1)

input_size = 1
nodes = 1
output_nodes = 10
net.add_layer(IdentityLayer(input_size, input_size, Identity))
net.add_layer(FullyConnectedLayer(input_size, nodes, Sigmoid))
net.add_layer(FullyConnectedLayer(nodes, output_nodes, Identity))

net.train()
test_accuracy = net.compute_test_accuracy()
print(f"Test accuracy: {test_accuracy}")









pass
"""




dataset_path = os.path.normpath(os.path.join(os.getcwd(), "../dataset/mnist/"))
data_loader = DataLoader(dataset_path, dataset_percentage=0.01, training_set_percentage=0.75) #loads mnist, splitting it into 75% training and 25% validation
data_loader.LoadDataset()

net = Network(data_loader, CrossEntropyWithSoftMax, learning_rate=0.1)

random.seed(3)

number_of_hidden_layers = 3
number_of_nodes = 6
number_of_output_nodes = 10
input_size = len(data_loader.train_X[0])

"""
input_layer = FullyConnectedLayer(input_size, 200, activation_function=Sigmoid)
net.add_layer(input_layer)

layer = FullyConnectedLayer(200, 80, activation_function=Sigmoid)
net.add_layer(layer)
layer = FullyConnectedLayer(80, 10, activation_function=Sigmoid)
net.add_layer(layer)
"""
input_layer = FullyConnectedLayer(input_size, 20, activation_function=Sigmoid)
net.add_layer(input_layer)
"""
for i in range(0, number_of_hidden_layers - 1):
    layer = FullyConnectedLayer(number_of_nodes, number_of_nodes, activation_function=Sigmoid)
    net.add_layer(layer)
"""
output_layer = FullyConnectedLayer(20, number_of_output_nodes, activation_function=Sigmoid)
net.add_layer(output_layer)

net.train(batch_size=1, MAX_EPOCH=5)

test_accuracy = net.compute_test_accuracy()
print(f"Test accuracy: {test_accuracy}")

print("Secondo me è un ", net.get_class(data_loader.train_X[10]))
image = np.reshape(data_loader.train_X[10], (28,28)) # 28 = sqrt(sample size)
fig = plt.figure
plt.imshow(image, cmap='gray')
plt.show()



