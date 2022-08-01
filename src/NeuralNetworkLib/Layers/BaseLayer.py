from abc import abstractmethod
from NeuralNetworkLib.ActivationFunctions.IFunction import IFunction
import numpy as np

class BaseLayer:

    def __init__(self, input_size: int, number_of_nodes: int, activation_function: IFunction) -> None:
        self.activation_function = activation_function
        self.input_size = input_size
        self.number_of_nodes = number_of_nodes
        
        self.__initialize_weight()
    

    def __initialize_weight(self, random_min=-0.01, random_max=0.01) -> None:
        W_size = (self.number_of_nodes, self.input_size + 1)
        self.W = np.random.uniform(random_min, random_max, W_size)
        
        self.delta = np.zeros(self.number_of_nodes)

    @abstractmethod
    def forward(self, X: np.array) -> np.array:
        pass
        