from abc import abstractmethod
from cmath import isnan
from datetime import datetime
import numpy as np
from NeuralNetworkLib.DataLoader import DataLoader
from NeuralNetworkLib.ErrorFunctions.CrossEntropyWithSoftMax import CrossEntropyWithSoftMax
from NeuralNetworkLib.Layers.BaseLayer import BaseLayer
from NeuralNetworkLib.StoppingCriteria.IStoppingCriterion import IStoppingCriterion
from NeuralNetworkLib.StoppingCriteria.NeverStopCriterion import NeverStopCriterion
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction

class BaseNetwork:

    def __init__(self, data_loader: DataLoader, error_function: IErrorFunction, stop_criterion: IStoppingCriterion = NeverStopCriterion) -> None:
        self.data_loader = data_loader
        self.stop_criterion = stop_criterion
        self.error_function = error_function

        self.train_X = data_loader.train_X
        self.train_Y = data_loader.train_Y
        self.validation_X = data_loader.validation_X
        self.validation_Y = data_loader.validation_Y
        self.test_X = data_loader.test_X
        self.test_Y = data_loader.test_Y

        self.training_error_history = []
        self.validation_error_history = []

        self.Layers = []

        self.cancel_flag = False


    def add_layer(self, layer: BaseLayer):
        self.Layers.append(layer)
    
    def train(self, batch_size, MAX_EPOCH=100):
        """Batch training process"""
        self.batch_size = batch_size

        if len(self.train_X) % self.batch_size != 0:
            print("Invalid batch size, should be a divisor of len(self.train_X)")
            return

        train_start_time = datetime.now()
        indici_training_set = np.arange(len(self.train_X))

        for epoch in range(0, MAX_EPOCH):
            epoch_start_time = datetime.now()

            np.random.shuffle(indici_training_set)

            n = 0
            while n < len(self.train_X):
                self.reset_error_derivative()
                for b in range(0, self.batch_size):
                    
                    x = self.train_X[indici_training_set[n]]
                    y = self.forward(x)
                    
                    t = self.train_Y[indici_training_set[n]]
                    self.backward(y, t)
                    self.update_derivative(x)
                    n += 1
                    
                self.update_weights()

                if self.cancel_flag:
                    break
            

            if self.cancel_flag:
                print("Training stopped")
                break

            training_error = self.compute_training_error()
            validation_error = self.compute_validation_error()

            if isnan(training_error) or isnan(validation_error):
                print("Error is invalid: break")
                break
            
            self.training_error_history.append(training_error)
            self.validation_error_history.append(validation_error)

            epoch_duration = datetime.now() - epoch_start_time

            print(f"Epoch #{epoch+1}: training error: {training_error}, validation error: {validation_error}. Took {epoch_duration}")

            if self.stop_criterion.should_stop(self.training_error_history, self.validation_error_history):
                print("Stopping criterion met.")
                break
            
        train_duration = datetime.now() - train_start_time
        print(f"Training completed in {train_duration}")

    def forward(self, x):
        tmp = x
        for layer in self.Layers:
            tmp = layer.forward(tmp)

        return tmp
    
    def reset_error_derivative(self):
        for layer in self.Layers:
            layer.delta.fill(0.0)
            layer.dW.fill(0.0)

    def backward(self, y, t):
        #calculate delta for the output layer
        output_layer = self.Layers[-1]

        for i in range(output_layer.number_of_nodes):
            output_layer.delta[i] = output_layer.activation_function.derivative(output_layer.activation[i]) * self.error_function.calculate_derivative(t[i], y[i]) 

        for i in reversed(range(len(self.Layers) - 1)):
            layer = self.Layers[i]
            next_layer = self.Layers[i+1]
            #g_prime_in_a = layer.activation_function.derivative(layer.activation)
            
            for j in range(layer.number_of_nodes): # for each neuron in the given (non-output) layer
                error = 0.0
                for k in range(next_layer.number_of_nodes): # for each neuron in the next layer
                    error += next_layer.W[k][j] * next_layer.delta[k]

                #error = np.sum(np.dot(next_layer.W[:,j], next_layer.delta))
                layer.delta[j] = error * layer.activation_function.derivative(layer.activation[j])

    def update_derivative(self, x):
        for l in range(len(self.Layers)):
            if l == 0:
                input = x
            else:
                input = self.Layers[l-1].output
            
            layer = self.Layers[l]
 
            input = np.append(input.copy(), 1.0)
            layer.dW += layer.delta.reshape(len(layer.delta), 1) @ input.reshape(1, len(input))

    @abstractmethod
    def update_weights(self):
        pass


    def get_class(self, x):
        y = self.forward(x)
        return np.argmax(y)

    def compute_training_error(self):
        """Returns the training error using the specified error function"""

        E = 0.0
        for n in range(0, len(self.train_X)):
            x = self.train_X[n]
            t = self.train_Y[n]
            y = self.forward(x)

            E += self.error_function.calculate(expected=t, actual=y)

        return E / len(self.train_X)

    def compute_validation_error(self):
        """Returns the validation error using the specified error function"""

        E = 0.0
        for n in range(0, len(self.validation_X)):
            x = self.validation_X[n]
            t = self.validation_Y[n]
            y = self.forward(x)

            E += self.error_function.calculate(expected=t, actual=y)

        return E/ len(self.validation_X)

    def compute_test_accuracy(self):
        """Returns the accuracy on test set"""

        correct_answers = 0

        for n in range (0, len(self.test_X)):
            x = self.test_X[n]
            t = self.test_Y[n]
            y_class = self.get_class(x)

            if (y_class == np.argmax(t)):
                correct_answers += 1

        return correct_answers / len(self.test_X)