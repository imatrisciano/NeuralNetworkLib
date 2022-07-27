from abc import abstractmethod


class IErrorFunction:
    
    @abstractmethod
    def calculate(expected, real):
        """Calcola il valore della funzione nel punto specificato"""
        pass