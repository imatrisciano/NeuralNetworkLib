import numpy as np

from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction
from NeuralNetworkLib.Layers.BaseLayer import BaseLayer

class FullyConnectedLayer(BaseLayer):
    
    def forward(self, X: np.array) -> np.array:
        self.input = X.copy()
        self.input = np.append(self.input, 1.0)
        self.unactivated_output = self.W @ self.input
        self.output = self.activation_function.calculate(self.unactivated_output)
        return self.output
        