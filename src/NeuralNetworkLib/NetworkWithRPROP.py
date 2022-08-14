import numpy as np
from NeuralNetworkLib.BaseNetwork import BaseNetwork
from NeuralNetworkLib.DataLoader import DataLoader
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction
from NeuralNetworkLib.StoppingCriteria.IStoppingCriterion import IStoppingCriterion
from NeuralNetworkLib.StoppingCriteria.NeverStopCriterion import NeverStopCriterion


class NetworkWithRPROP(BaseNetwork):
    """Implements a Neural Network using the Resilient Backpropagation (RPROP)."""

    def __init__(self, data_loader: DataLoader, error_function: IErrorFunction, stop_criterion: IStoppingCriterion = NeverStopCriterion, eta_pos=1.2, eta_neg=0.5) -> None:

        super().__init__(data_loader, error_function, stop_criterion) #call base class constructor

        self.eta_pos = eta_pos
        self.eta_neg = eta_neg
        self.rprop_delta_max = 50.0
        self.rprop_delta_min = 1e-6
        self.delta_initial_value = 0.01 #we initialize rprop_deltas to a small value
    
    def init_train(self):
        super().init_train()
        for layer in self.Layers:
            layer.dW_old = np.zeros_like(layer.W)
            layer.rprop_deltas = np.empty_like(layer.W)
            layer.rprop_deltas.fill(self.delta_initial_value) 

    def update_weights(self):
        for layer in self.Layers:
            #we now compute the sign of the element-wise multiplication between dW and dW_old 
            #this gives us a matrix with the same shape as dW
            #each cell will be > 0 if dW and dW_old have the same sign, < 0 if they have opposite sign or 0 if one of them is 0
            dw_sign = np.multiply(layer.dW, layer.dW_old)
            
            #we now iterate over every element of that matrix
            for i in range(layer.number_of_nodes):
                for j in range(layer.input_size + 1):
                    if (dw_sign[i][j] > 0): #if the error kept the same sign
                        layer.rprop_deltas[i][j] = min(layer.rprop_deltas[i][j] * self.eta_pos, self.rprop_delta_max) #update the rprop_delta, increasing it by eta_pos until we hit rprop_delta_max
                        layer.W[i][j] -= np.sign(layer.dW[i][j]) * layer.rprop_deltas[i][j] #update the weight
                        layer.dW_old[i][j] = layer.dW[i][j] #update the old error
                    elif dw_sign[i][j] < 0 : #if the error changed sign
                        layer.rprop_deltas[i][j] = max(layer.rprop_deltas[i][j] * self.eta_neg, self.rprop_delta_min) #decrease rprop_delta until we hit rprop_delta_min
                        layer.dW_old[i][j] = 0 #zero-out the old error
                    else: #if the error was 0
                        layer.W[i][j] -= np.sign(layer.dW[i][j]) * layer.rprop_deltas[i][j] #update the weight
                        layer.dW_old[i][j] = layer.dW[i][j] #update the old error