import numpy as np
from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction


class LeakyReLU(IFunction):
    def calculate(x):
        return x*np.heaviside(x, 1) + 1/16 * x * (1-np.heaviside(x, 1))
        

    def derivative(x):
        return np.heaviside(x, 1) + 1/16 * (1-np.heaviside(x, 1))
