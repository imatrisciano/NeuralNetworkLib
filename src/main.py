from NeuralNetworkLib.Network import Network
from NeuralNetworkLib.ActivationFunctions.ReLU import ReLU
from NeuralNetworkLib.DataLoader import DataLoader
from NeuralNetworkLib.Layers.FullyConnectedLayer import FullyConnectedLayer



data_loader = DataLoader("../dataset/mnist", training_set_percentage=0.75) #loads mnist, splitting it into 75% training and 25% validation
data_loader.LoadDataset()

net = Network(data_loader)

number_of_layers = 3
number_of_nodes = 100
number_of_output_nodes = 10
input_size = len(data_loader.train_X[0])

for i in range(0, number_of_layers):
    layer = FullyConnectedLayer(input_size, number_of_nodes, activation_function=ReLU)
    net.add_layer(layer)

output_layer = FullyConnectedLayer(input_size, number_of_output_nodes, activation_function=ReLU)
net.add_layer(output_layer)



