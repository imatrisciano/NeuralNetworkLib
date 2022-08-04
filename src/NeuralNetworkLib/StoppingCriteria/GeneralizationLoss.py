import numpy as np
from NeuralNetworkLib.StoppingCriteria.IStoppingCriterion import IStoppingCriterion


class GeneralizationLoss(IStoppingCriterion):
    def __init__(self, alpha):
        self.alpha = alpha

    def should_stop(self, training_error_history, validation_error_history) -> bool:
        should_stop = GeneralizationLoss.GL(validation_error_history) > self.alpha

        return should_stop
    
    def GL(validation_error_history):
        best_validation_error = np.min(validation_error_history) #the lowest validation error so far, aka E_opt
        last_validation_error = validation_error_history[-1] #aka E_va

        GL = 100 * (last_validation_error / best_validation_error - 1)
        return GL
