import numpy as np

from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction
from NeuralNetworkLib.Layers.BaseLayer import BaseLayer

class FullyConnectedLayer(BaseLayer):
    
    def forward(self, X: np.array) -> np.array:
        return self.activation_function(self.W * X)
        



