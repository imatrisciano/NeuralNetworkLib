from numpy import log
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction


class CrossEntropy(IErrorFunction):
    def calculate(expected, real):
        return real * log(expected)