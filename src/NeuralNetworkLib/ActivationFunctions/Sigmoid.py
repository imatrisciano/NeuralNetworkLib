import numpy as np
from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction


class Sigmoid(IFunction):
    def calculate(x):
        #ones = np.ones_like(x)
        return 1 / (1 + np.exp(-x))

    def derivative(x):
        z = Sigmoid.calculate(x)
        return z * (1 - z)