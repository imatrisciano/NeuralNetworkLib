from datetime import datetime
import time
import numpy as np
from NeuralNetworkLib.DataLoader import DataLoader
from NeuralNetworkLib.Layers.BaseLayer import BaseLayer
from NeuralNetworkLib.StoppingCriteria.IStoppingCriterion import IStoppingCriterion
from NeuralNetworkLib.StoppingCriteria.NeverStopCriterion import NeverStopCriterion
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction

class Network:

    def __init__(self, data_loader: DataLoader, error_function: IErrorFunction, MAX_EPOCH=1000, stop_criterion: IStoppingCriterion = NeverStopCriterion) -> None:
        self.data_loader = data_loader
        self.stop_criterion = stop_criterion
        self.MAX_EPOCH = MAX_EPOCH
        self.error_function = error_function

        self.train_X, self.train_Y, self.validation_X, self.validation_Y, self.test_X, self.test_Y = data_loader.LoadDataset()

        self.training_error_history = []
        self.validation_error_history = []

        self.Layers = []


    def add_layer(self, layer: BaseLayer):
        self.Layers.append(layer)
    
    def train(self):
        """Batch training process"""

        train_start_time = datetime.now()
        old_training_loss = 0.0

        for epoch in range(0, self.MAX_EPOCH):
            
            epoch_start_time = time()

            if self.stop_criterion.should_stop(self.training_error_history, self.validation_error_history):
                print("Stopping criterion met.")
                break

            self.reset_error_derivative()

            for n in range(0, len(self.train_X)):
                x = self.train_X[n]
                y = self.forward(x)
                training_loss = self.compute_training_error()

                self.backward(y, training_loss, old_training_loss)
                self.update_error_derivative()
            
            old_training_loss = training_loss
            self.update_weights()

            training_error = self.compute_training_error()
            validation_error = self.compute_validation_error()

            self.training_error_history.append(training_error)
            self.validation_error_history.append(validation_error)

            epoch_duration = time() - epoch_start_time

            print(f"Epoch #{epoch}: training error: {training_error}, validation error: {validation_error}. Took {1000.0 * epoch_duration} ms")
        
        train_duration = datetime.now() - train_start_time
        print(f"Training completed in {train_duration}")

    def forward(self, x):
        for layer in self.Layers:
            x = layer.forward(x)
        return x
        

    def backward(self):
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

            E += self.error_function.calculate(expected=t, real=y)

        return E

    def compute_validation_error(self):
        """Returns the validation error using the specified error function"""

        E = 0.0
        for n in range(0, len(self.validation_X)):
            x = self.validation_X[n]
            t = self.validation_Y[n]
            y = self.forward(x)

            E += self.error_function.calculate(expected=t, real=y)

        return E

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