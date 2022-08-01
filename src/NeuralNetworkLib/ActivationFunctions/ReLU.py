import numpy as np
from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction


class ReLU(IFunction):
    def calculate(x):
        return x*np.heaviside(x, 0.5) #max(0, x) = x*heaviside(x) = x*(np.sign(x) + 1.0) / 2.0

    def derivative(x):
        return np.heaviside(x, 0.5) #(np.sign(x) + 1.0) / 2.0 # heaviside
