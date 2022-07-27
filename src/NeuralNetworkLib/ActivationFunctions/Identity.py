from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction


class Identity(IFunction):
    def calculate(x):
        return x
