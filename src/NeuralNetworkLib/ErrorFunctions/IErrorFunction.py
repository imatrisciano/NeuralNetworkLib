from abc import abstractmethod


class IErrorFunction:
    
    @abstractmethod
    def calculate(expected, actual):
        pass

    @abstractmethod
    def calculate_derivative(expected, actual):
        pass