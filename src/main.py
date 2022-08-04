from cgi import test
import random
import signal
from NeuralNetworkLib.ActivationFunctions.LeakyReLU import LeakyReLU

from NeuralNetworkLib.ErrorFunctions.CrossEntropyWithSoftMax import CrossEntropyWithSoftMax
from NeuralNetworkLib.Layers.IdentityLayer import IdentityLayer
from NeuralNetworkLib.SimpleNetwork import SimpleNetwork
from NeuralNetworkLib.NetworkWithRPROP import NetworkWithRPROP
from NeuralNetworkLib.ActivationFunctions.ReLU import ReLU
from NeuralNetworkLib.ActivationFunctions.Sigmoid import Sigmoid
from NeuralNetworkLib.ActivationFunctions.Identity import Identity
from NeuralNetworkLib.DataLoader import DataLoader
from NeuralNetworkLib.Layers.FullyConnectedLayer import FullyConnectedLayer
from NeuralNetworkLib.ErrorFunctions.CrossEntropy import CrossEntropy
import os
import matplotlib.pyplot as plt
import numpy as np

from NeuralNetworkLib.StoppingCriteria.GeneralizationLoss import GeneralizationLoss
from NeuralNetworkLib.StoppingCriteria.ProgressQuotient import ProgressQuotient


"""
learning 0.15, input->20->output, batch 1, training 450,  5 epoch   : test 0.4494
learning 0.15, input->20->output, batch 1, training 450, 15 epoch   : test 0.5045
learning 0.15, input->20->output, batch 1, training 450,  5 epoch  con softmax  : test 0.4494


learning 0.15, input->20->output, batch 10, training 450   : test 0.schifo
learning 0.15, input->40->20->output, batch 1, training 450   : test 0.5

learning 0.15, input->20->output, batch 1, training 450,  5 epoch, pesi in [-0.1, 0.1]   : test 0.5
learning 0.15, input->20->output, batch 1, training 0.5,  10 epoch, pesi in [-0.1, 0.1]   : test 0.9153 #forse il batch era a 10
learning 0.15, input->20->output, batch 1, training 0.3,  10 epoch, pesi in [-0.1, 0.1]   : test 0.9217
learning 0.05, input->20->output, batch 1, training 0.3,  10 epoch, pesi in [-0.1, 0.1]   : test 0.9203
learning 0.25, input->20->output, batch 1, training 0.3,  10 epoch, pesi in [-0.1, 0.1]   : test 0.9131


eta_pos=1.2, eta_neg=0.5, input->20->output, batch 200, training 0.3, 5 epoch  : test 0.8487 
eta_pos=1.2, eta_neg=0.5, input->20->output, batch 200, training 0.3, 3 epoch  : test 0.8388
eta_pos=1.2, eta_neg=0.5, input->20->output, batch 200, training 1, 3 epoch  : test 0.853
eta_pos=1.2, eta_neg=0.5, input->20->output, batch 200, training 1, 5 epoch  : test 0.8708
"""



dataset_path = os.path.normpath(os.path.join(os.getcwd(), "../dataset/mnist/"))
data_loader = DataLoader(dataset_path, dataset_percentage=0.1, training_set_percentage=0.75, test_set_size=10000) #loads mnist, splitting it into 75% training and 25% validation
data_loader.LoadDataset()


#stop_criterion = GeneralizationLoss(alpha=0.1)
stop_criterion = ProgressQuotient(alpha=0.02, strip_length=5)


#https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=FB6D482F7A3F9863F0003732B12BD8D9?doi=10.1.1.21.3428&rep=rep1&type=pdf
net = SimpleNetwork(data_loader, CrossEntropyWithSoftMax, learning_rate=0.15, stop_criterion=stop_criterion)
#net = NetworkWithRPROP(data_loader, CrossEntropyWithSoftMax, eta_pos=1.2, eta_neg=0.5, stop_criterion=stop_criterion)


def ctrl_c_handler(signum, frame):
    net.cancel_flag = True

signal.signal(signal.SIGINT, ctrl_c_handler)

random.seed(3)

number_of_hidden_layers = 3
number_of_nodes = 6
number_of_output_nodes = 10
input_size = len(data_loader.train_X[0])

net.add_layer(FullyConnectedLayer(input_size, 20, activation_function=Sigmoid))
net.add_layer(FullyConnectedLayer(net.Layers[-1].number_of_nodes, number_of_output_nodes, activation_function=Sigmoid))

net.train(batch_size=100, MAX_EPOCH=100)

test_accuracy = net.compute_test_accuracy()
print(f"Test accuracy: {test_accuracy}")

"""
print("Secondo me è un", net.get_class(data_loader.train_X[10]))
image = np.reshape(data_loader.train_X[10], (28,28)) # 28 = sqrt(sample size)
fig = plt.figure
plt.imshow(image, cmap='gray')
plt.show()
"""

time = range(1, len(net.training_error_history) + 1)
plt.plot(time, net.training_error_history, marker="o")
plt.plot(time, net.validation_error_history, marker="o")

plt.title("Training and validation error history")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend(["Training error","Validation error"])
plt.show()
