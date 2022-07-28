from abc import abstractmethod


class IFunction:
    
    @abstractmethod
    def calculate(x):
        pass