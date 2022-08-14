import numpy as np
from NeuralNetworkLib.BaseNetwork import BaseNetwork
from NeuralNetworkLib.DataLoader import DataLoader
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction
from NeuralNetworkLib.StoppingCriteria.IStoppingCriterion import IStoppingCriterion
from NeuralNetworkLib.StoppingCriteria.NeverStopCriterion import NeverStopCriterion


class SimpleNetwork(BaseNetwork):
    def __init__(self, data_loader: DataLoader, error_function: IErrorFunction, stop_criterion: IStoppingCriterion = NeverStopCriterion, learning_rate = 0.1) -> None:
        super().__init__(data_loader, error_function, stop_criterion)
        self.learning_rate = learning_rate

    def update_weights(self):
        for layer in self.Layers:
            layer.W -= self.learning_rate * layer.dW