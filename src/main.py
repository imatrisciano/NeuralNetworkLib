from queue import Full
from NeuralNetworkLib.Network import Network
from NeuralNetworkLib.ActivationFunctions.ReLU import ReLU
from NeuralNetworkLib.DataLoader import DataLoader
from NeuralNetworkLib.Layers.FullyConnectedLayer import FullyConnectedLayer



data_loader = DataLoader()

net = Network(data_loader)

train_X = ...

number_of_layers = 3
number_of_nodes = 10
input_size = len(train_X[0])

for i in range(0, number_of_layers):
    layer = FullyConnectedLayer(input_size, number_of_nodes, activation_function=ReLU)
    net.add_layer(layer)





