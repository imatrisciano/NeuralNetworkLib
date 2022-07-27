from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction


class SumOfSquares(IErrorFunction):
    def calculate(expected, real):
        return (expected - real) ** 2