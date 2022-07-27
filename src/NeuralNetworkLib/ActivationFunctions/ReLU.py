from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction


class ReLU(IFunction):
    def calculate(x):
        return max(0, x)
