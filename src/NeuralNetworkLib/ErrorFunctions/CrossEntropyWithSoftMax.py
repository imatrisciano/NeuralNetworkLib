import numpy as np
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction


class CrossEntropyWithSoftMax(IErrorFunction):
    def calculate(expected, actual):
        sum = np.sum(np.exp(actual))
        z = np.exp(actual) / sum
        return - np.sum(expected * np.log(z))

    def calculate_derivative(expected, actual):
        return actual - expected