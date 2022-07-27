
from NeuralNetworkLib.StoppingCriteria.IStoppingCriterion import IStoppingCriterion


class NeverStopCriterion(IStoppingCriterion):
    def should_stop(training_error_history, validation_error_history) -> bool:
        return False
