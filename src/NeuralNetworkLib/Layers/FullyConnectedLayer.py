import numpy as np

from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction
from NeuralNetworkLib.Layers.BaseLayer import BaseLayer

class FullyConnectedLayer(BaseLayer):
    
    def forward(self, X: np.array) -> np.array:
        X.append(1.0) #adding the bias
        self.output = self.activation_function(self.W * X)
        return self.output
        



