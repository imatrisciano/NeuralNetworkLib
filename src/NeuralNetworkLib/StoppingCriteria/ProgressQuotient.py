import numpy as np
from NeuralNetworkLib.StoppingCriteria.GeneralizationLoss import GeneralizationLoss
from NeuralNetworkLib.StoppingCriteria.IStoppingCriterion import IStoppingCriterion


class ProgressQuotient(IStoppingCriterion):
    def __init__(self, alpha, strip_length):
        self.alpha = alpha
        self.strip_length = strip_length # aka k

    

    def should_stop(self, training_error_history, validation_error_history) -> bool:
        GL = GeneralizationLoss.GL(validation_error_history)
        P_k = ProgressQuotient.training_progress(self.strip_length, training_error_history)

        PQ = GL/P_k

        should_stop = PQ > self.alpha
        return should_stop

    def training_progress(strip_length, training_error_history):
        """how much was the average training error during the strip larger than the minimum training error during the strip?"""

        strip_start = max(len(training_error_history) - strip_length, 0) # if there are not enough epochs, the strip_length is the biggest possible
        strip_end = len(training_error_history)

        strip = training_error_history[strip_start:strip_end]
        strip_error_sum = np.sum(strip)
        strip_min_error = np.min(strip)

        P_k = 1000 * (strip_error_sum/(len(strip) * strip_min_error) - 1)

        return P_k


