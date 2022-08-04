import numpy as np
from NeuralNetworkLib.BaseNetwork import BaseNetwork
from NeuralNetworkLib.DataLoader import DataLoader
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction
from NeuralNetworkLib.StoppingCriteria.IStoppingCriterion import IStoppingCriterion
from NeuralNetworkLib.StoppingCriteria.NeverStopCriterion import NeverStopCriterion


class NetworkWithRPROP(BaseNetwork):
    def __init__(self, data_loader: DataLoader, error_function: IErrorFunction, stop_criterion: IStoppingCriterion = NeverStopCriterion, eta_pos=1.2, eta_neg=0.5) -> None:
        super().__init__(data_loader, error_function, stop_criterion)
        self.eta_pos = eta_pos
        self.eta_neg = eta_neg
        self.eta_max = 50.0
        self.eta_min = 1e-6
        self.first_epoch = True
    
    def update_weights(self):
        if self.first_epoch:
            for layer in self.Layers:
                layer.dW_old = np.zeros_like(layer.W)
                layer.rprop_deltas = np.ones_like(layer.W)
                layer.rprop_deltas.fill(1e-3)

            self.first_epoch = False

        for layer in self.Layers:
            dw_sign = np.sign(np.multiply(layer.dW, layer.dW_old))

            for i in range(len(layer.W)):
                for j in range(len(layer.W[i])):
                    if (dw_sign[i][j] > 0):
                        layer.rprop_deltas[i][j] = max(layer.rprop_deltas[i][j] * self.eta_pos, self.eta_min)
                        layer.W[i][j] -= np.sign(layer.dW[i][j]) * layer.rprop_deltas[i][j]
                        layer.dW_old[i][j] = layer.dW[i][j]
                    elif dw_sign[i][j] < 0 :
                        layer.rprop_deltas[i][j] = min(layer.rprop_deltas[i][j] * self.eta_neg, self.eta_max)
                        layer.dW_old[i][j] = 0
                    else:
                        layer.W[i][j] -= np.sign(layer.dW[i][j]) * layer.rprop_deltas[i][j]
                        layer.dW_old[i][j] = layer.dW[i][j]



            """   


            
            dW_change = np.sign(np.multiply(layer.dW, layer.dW_old))
            layer.rprop_deltas = np.multiply(layer.rprop_deltas, self.get_etas(dW_change))

            layer.W -= np.multiply(layer.rprop_deltas, layer.dW)
            layer.dW_old = layer.dW.copy()
            """
            
    
    def get_etas(self, dW_change):
        return self.eta_pos * np.heaviside(dW_change, 0) + self.eta_neg * (np.ones_like(dW_change) - np.heaviside(dW_change, 1))
