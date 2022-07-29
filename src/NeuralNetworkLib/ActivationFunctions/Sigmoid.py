import numpy as np
from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction


class Sigmoid(IFunction):
    def calculate(x):
        return 1.0/(1.0 + np.exp(-x))

    def derivative(x):
        z = Sigmoid.calculate(x)
        return z * (1.0 - z)