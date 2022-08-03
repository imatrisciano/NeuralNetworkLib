import numpy as np

from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction
from NeuralNetworkLib.Layers.BaseLayer import BaseLayer

class FullyConnectedLayer(BaseLayer):
    
    def forward(self, X: np.array) -> np.array:
        input = np.array(X, copy=True)
        input = np.append(input, 1.0)
        input = input.reshape(len(input), 1)



        self.activation = self.W @ input
        self.output = self.activation_function.calculate(self.activation)
        return self.output
        """
        self.activation = np.zeros(self.number_of_nodes)
        for i in range(self.number_of_nodes): # does not include the last one, which is bias
            for j in range(self.input_size - 1):
                self.activation[i] += self.W[i][j] * X[j] # weighted sum plus bias
            self.activation[i] += self.W[i][-1]
            self.output[i] = self.activation_function.calculate(self.activation[i])
        return self.output
        """
        
        

        """
        self.input = X
        self.activation = self.W @ self.input + self.bias
        self.output = self.activation_function.calculate(self.unactivated_output)
        return self.output
        """
        