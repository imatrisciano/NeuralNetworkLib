import numpy as np
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction


class CrossEntropy(IErrorFunction):
    def calculate(expected, actual):
        return np.sum(expected * np.log(actual))

    def calculate_derivative(expected, actual):
        return 