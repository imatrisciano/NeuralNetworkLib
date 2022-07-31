import numpy as np
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction


class MeanSquareError(IErrorFunction):
    def calculate(expected, actual):
        return np.mean((expected - actual)**2)

    def calculate_derivative(expected, actual):
        return (1 / expected.shape[0]) * (1/expected.shape[-1]) * -2 * np.sum(actual - expected, axis=0)

