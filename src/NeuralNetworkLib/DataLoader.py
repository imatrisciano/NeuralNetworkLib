from mnist import MNIST #pip install python-mnist

class DataLoader:

    def __init__(self, dataset_path, training_set_percentage = 0.75):
        self.dataset_path = dataset_path
        self.training_set_percentage = training_set_percentage

    def LoadDataset(self):
        """ Loads the dataset and splits it into training and validation. Also loads the test set.
        Training set size: dataset_size * training_set_percentage
        Validation set size: dataset_size * (1-training_set_percentage)"""

        mndata = MNIST(self.dataset_path)

        set_X, set_Y = mndata.load_training()

        training_set_size = len(set_X) * self.training_set_percentage
        validation_set_size = len(set_X) - training_set_size

        self.train_X = set_X[0: training_set_size]
        self.train_Y = set_Y[0: training_set_size]

        self.validation_X = set_X[training_set_size : validation_set_size]
        self.validation_Y = set_Y[training_set_size : validation_set_size]
        
        self.test_X, self.test_Y = mndata.load_testing()