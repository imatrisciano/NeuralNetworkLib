from abc import abstractmethod


class IFunction:
    
    @abstractmethod
    def calculate(x):
        """Calcola il valore della funzione nel punto specificato"""
        pass