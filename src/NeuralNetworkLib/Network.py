from cmath import isnan
from datetime import datetime
import time
import numpy as np
from NeuralNetworkLib.DataLoader import DataLoader
from NeuralNetworkLib.Layers.BaseLayer import BaseLayer
from NeuralNetworkLib.StoppingCriteria.IStoppingCriterion import IStoppingCriterion
from NeuralNetworkLib.StoppingCriteria.NeverStopCriterion import NeverStopCriterion
from NeuralNetworkLib.ErrorFunctions.IErrorFunction import IErrorFunction

class Network:

    def __init__(self, data_loader: DataLoader, error_function: IErrorFunction, stop_criterion: IStoppingCriterion = NeverStopCriterion, learning_rate = 0.1) -> None:
        self.data_loader = data_loader
        self.stop_criterion = stop_criterion
        self.error_function = error_function
        self.learning_rate = learning_rate

        self.train_X = data_loader.train_X
        self.train_Y = data_loader.train_Y
        self.validation_X = data_loader.validation_X
        self.validation_Y = data_loader.validation_Y
        self.test_X = data_loader.test_X
        self.test_Y = data_loader.test_Y

        self.training_error_history = []
        self.validation_error_history = []

        self.batch_size = len(self.train_X)
        self.Layers = []


    def add_layer(self, layer: BaseLayer):
        self.Layers.append(layer)
    
    def train(self, MAX_EPOCH=100):
        """Batch training process"""

        train_start_time = datetime.now()

        for epoch in range(0, MAX_EPOCH):
            epoch_start_time = time.time()

            if self.stop_criterion.should_stop(self.training_error_history, self.validation_error_history):
                print("Stopping criterion met.")
                break

            
            self.reset_error_derivative()
            for n in range(0, self.batch_size):
                
                x = self.train_X[n]
                y = self.forward(x)
                
                t = self.train_Y[n]
                self.backward(y, t)
                self.update_derivative(x)
                
            self.update_weights()

            

            training_error = self.compute_training_error()
            validation_error = self.compute_validation_error()

            if isnan(training_error) or isnan(validation_error):
                print("Error is invalid: break")
                break
            
            self.training_error_history.append(training_error)
            self.validation_error_history.append(validation_error)

            epoch_duration = time.time() - epoch_start_time

            print(f"Epoch #{epoch}: training error: {training_error}, validation error: {validation_error}. Took {1000.0 * epoch_duration} ms")
            
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

        #a = output_layer.activation_function.derivative(output_layer.activation)
        #b = self.error_function.calculate_derivative(expected=t, actual=y) 
        for i in range(output_layer.number_of_nodes):
            #output_layer.delta[i] = a[i]*b[i]
            output_layer.delta[i] = output_layer.activation_function.derivative(output_layer.activation[i]) * self.error_function.calculate_derivative(t[i], y[i]) 
        #output_layer.delta += a*b

        """
        for i in range(len(self.Layers) - 2, 0, -1):
            layer = self.Layers[i]
            next_layer = self.Layers[i+1]
            ##layer.delta += layer.activation_function.derivative(layer.unactivated_output) * np.dot(next_layer.delta.reshape(1,-1), next_layer.W)
            #layer.delta = layer.activation_function.derivative(layer.output) * (next_layer.W * next_layer.delta)
           
            
            g_prime_in_a = layer.activation_function.derivative(layer.unactivated_output)
            for h in range (0, layer.number_of_nodes):
                sum = 0.0
                for k in range(0, next_layer.number_of_nodes):
                    w = next_layer.W[k, h]
                    delta_k = next_layer.delta[k]
                    sum += w*delta_k
                layer.delta[h] += g_prime_in_a[h] * sum 
        """
        for i in reversed(range(len(self.Layers) - 1)):
            layer = self.Layers[i]
            next_layer = self.Layers[i+1]
            #g_prime_in_a = layer.activation_function.derivative(layer.activation)
            
            for j in range(layer.number_of_nodes): # for each neuron in the given (non-output) layer
                error = 0.0
                for k in range(next_layer.number_of_nodes): # for each neuron in the next layer
                    error += next_layer.W[k][j] * next_layer.delta[k]
                layer.delta[j] = error * layer.activation_function.derivative(layer.activation[j])

            #layer.delta += layer.activation_function.derivative(layer.W @ prev_layer.output) * np.dot(next_layer.delta, next_layer.W)
        
    def update_derivative(self, x):
        for l in range(len(self.Layers)):
            if l == 0:
                input = x
            else:
                input = self.Layers[l-1].output
            
            layer = self.Layers[l]
 
            for i in range(layer.number_of_nodes):
                for j in range(layer.input_size):
                    layer.dW[i][j] += layer.delta[i] * input[j] 
                    layer.dW[i][-1] += layer.delta[i] # * 1 : bias

    def update_weights(self):
        for layer in self.Layers:
            layer.W -= self.learning_rate /self.batch_size * layer.dW
        """
        for layer in self.Layers:
            for i in range(layer.number_of_nodes):
                for j in range(layer.input_size ):
                    layer.W[i][j] -= self.learning_rate * 1.0/self.batch_size * layer.dW[i][j]
        """


        return
        for l in range (0, len(self.Layers)):
            layer = self.Layers[l]
            #for j in range(0, len(layer.W)):
            """
            for n in range(0, layer.number_of_nodes):
                delta_neurone = layer.delta[n]
                input_neurone = layer.input[n] #??
                #https://brilliant.org/wiki/backpropagation/ 
                #TODO
            """
            for i in range(0, len(layer.W)):
                for j in range (0, len(layer.W[0])):
                    dW = layer.delta[i] * layer.total_input[j] / self.batch_size
                    layer.W[i][j] -= self.learning_rate * dW

            """
            delta_T = np.reshape(layer.delta, (len(layer.delta), 1))
            dW = (delta_T @ np.reshape(layer.input, (1, len(layer.input)))) / self.batch_size
            layer.W -= self.learning_rate * dW
            """


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