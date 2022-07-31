import numpy as np
from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction


class ReLU(IFunction):
    def calculate(x):
        return x*(np.sign(x) + 1.0) / 2.0 #max(0, x) = x*heaviside(x)

    def derivative(x):
        return (np.sign(x) + 1.0) / 2.0 # heaviside
