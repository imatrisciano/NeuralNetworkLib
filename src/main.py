from NeuralNetworkLib.Network import Network
from NeuralNetworkLib.ActivationFunctions.ReLU import ReLU
from NeuralNetworkLib.DataLoader import DataLoader
from NeuralNetworkLib.Layers.FullyConnectedLayer import FullyConnectedLayer
from NeuralNetworkLib.ErrorFunctions.CrossEntropy import CrossEntropy
import os
import matplotlib.pyplot as plt
import numpy as np

dataset_path = os.path.normpath(os.path.join(os.getcwd(), "../dataset/mnist/"))
data_loader = DataLoader(dataset_path, training_set_percentage=0.75) #loads mnist, splitting it into 75% training and 25% validation
data_loader.LoadDataset()

net = Network(data_loader, CrossEntropy)

number_of_hidden_layers = 3
number_of_nodes = 10
number_of_output_nodes = 10
input_size = len(data_loader.train_X[0])

input_layer = FullyConnectedLayer(input_size + 1, number_of_nodes, activation_function=ReLU)
net.add_layer(input_layer)

for i in range(0, number_of_hidden_layers):
    layer = FullyConnectedLayer(number_of_nodes, number_of_nodes, activation_function=ReLU)
    net.add_layer(layer)

output_layer = FullyConnectedLayer(number_of_nodes, number_of_output_nodes, activation_function=ReLU)
net.add_layer(output_layer)

net.train(MAX_EPOCH=10)
print(net.get_class(data_loader.train_X[1]))
image = np.reshape(data_loader.train_X[1], (28,28)) # 28 = sqrt(sample size)
fig = plt.figure
plt.imshow(image, cmap='gray')
plt.show()

