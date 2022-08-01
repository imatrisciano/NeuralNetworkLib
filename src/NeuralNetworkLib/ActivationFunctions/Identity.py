from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction
import numpy as np

class Identity(IFunction):
    def calculate(x):
        return x

    def derivative(x):
        return np.ones_like(x)
