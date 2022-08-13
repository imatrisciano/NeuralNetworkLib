from mnist import MNIST #pip install python-mnist
import numpy as np

class DataLoader:

    def __init__(self, dataset_path, dataset_percentage = 0.3, training_set_percentage = 0.75, test_set_size = -1):
        assert (dataset_percentage > 0 and dataset_percentage <= 1), "dataset_percentage can only be in range (0,1]"
        assert (training_set_percentage > 0 and training_set_percentage <= 1), "training_set_percentage can only be in range (0,1]"
        
        self.dataset_path = dataset_path
        self.training_set_percentage = training_set_percentage
        self.dataset_percentage = dataset_percentage
        self.test_set_size = test_set_size

    def LoadDataset(self):
        """ Loads the dataset and splits it into training and validation. Also loads the test set.
        Training set size: dataset_size * training_set_percentage
        Validation set size: dataset_size * (1-training_set_percentage)"""

        print("Loading dataset...")
        mndata = MNIST(self.dataset_path)
        set_X, set_Y = mndata.load_training()


        #split the dataset into training and validation sets
        training_set_size = int (len(set_X) * self.training_set_percentage * self.dataset_percentage)
        validation_set_size = int (len(set_X)*self.dataset_percentage - training_set_size)

        train_X = set_X[0: training_set_size]
        train_Y = set_Y[0: training_set_size]

        validation_X = set_X[training_set_size: training_set_size + validation_set_size]
        validation_Y = set_Y[training_set_size: training_set_size + validation_set_size]
        

        #load the test set
        test_X, test_Y = mndata.load_testing()
        self.labels = np.unique(train_Y)


        #preparing training set data
        self.train_X = np.array(train_X) / 255.0

        self.train_Y = self.labels_to_one_hot(train_Y)

        print(f"Training set loaded: {len(self.train_X)} elements")


        # preparing validation set data
        self.validation_X = np.array(validation_X) / 255
        self.validation_Y = self.labels_to_one_hot(validation_Y)

        print(f"Validation set loaded: {len(self.validation_X)} elements")


        # preparing test set data
        if self.test_set_size != -1 and self.test_set_size <= len(test_X):
            # if requested reduce test set size for performance reasons
            test_X = test_X[0:self.test_set_size]
            test_Y = test_Y[0:self.test_set_size]

        self.test_X = np.array(test_X) / 255
        self.test_Y = self.labels_to_one_hot(test_Y)

        print(f"Test set loaded: {len(self.test_X)} elements")

    def labels_to_one_hot(self, Y) -> np.array:
        one_hot = np.empty( (len(Y), len(self.labels), 1) )
        for i in range(0, len(Y)):
            a = (np.eye(len(self.labels))[Y[i]])
            one_hot[i] = a.reshape(len(a), 1)
        return one_hot
    
    def label_to_one_hot(self, label):
        return np.eye(len(self.labels))[label]