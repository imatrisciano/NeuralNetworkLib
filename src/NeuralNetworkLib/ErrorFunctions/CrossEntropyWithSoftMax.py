import numpy as np
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction


class CrossEntropyWithSoftMax(IErrorFunction):
    def calculate(expected, actual):
        """
        sum = np.sum(np.exp(actual))
        z = np.exp(actual) / sum
        return - np.sum(expected * np.log(z))
        """
        sum_actual = 0.0
        for i in range(len(actual)):
            sum_actual += np.exp(actual[i])

        real_label_index = np.argmax(actual)
        sum = - np.log(np.exp(actual[real_label_index]) / sum_actual )
        return sum


    def calculate_derivative(expected, actual):
        return actual - expected