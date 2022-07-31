import numpy as np

from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction
from NeuralNetworkLib.Layers.BaseLayer import BaseLayer

class FullyConnectedLayer(BaseLayer):
    
    def forward(self, X: np.array) -> np.array:
        self.output = self.activation_function.calculate(np.dot(X, self.W))
        return self.output
        



