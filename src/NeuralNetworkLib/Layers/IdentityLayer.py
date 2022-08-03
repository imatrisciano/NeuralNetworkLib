import numpy as np

from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction
from NeuralNetworkLib.Layers.BaseLayer import BaseLayer

class IdentityLayer(BaseLayer):
    
    def forward(self, X: np.array) -> np.array:
        self.input = X.copy()
        self.input = np.append(self.input, 1.0)
        self.unactivated_output = X
        self.output = self.unactivated_output
        return self.output
        