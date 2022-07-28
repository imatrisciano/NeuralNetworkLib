import numpy as np
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction


class SumOfSquares(IErrorFunction):
    def calculate(expected, actual):
        return np.sum((actual - expected) ** 2)
