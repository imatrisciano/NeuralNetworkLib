from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction


class Step(IFunction):
    def calculate(x):
        if x >= 0:
            return 1
        else:
            return 0
