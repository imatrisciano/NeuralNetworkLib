import numpy as np
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction


class CrossEntropyWithSoftMax(IErrorFunction):
    def calculate(expected, actual):
        sum_actual = np.sum(np.exp(actual))

        real_label_index = np.argmax(actual)
        sum = - np.log(np.exp(actual[real_label_index]) / sum_actual )
        return sum


    def calculate_derivative(expected, actual):
        return actual - expected