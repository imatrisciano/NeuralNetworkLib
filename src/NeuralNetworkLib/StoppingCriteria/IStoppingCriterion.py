from abc import abstractmethod


class IStoppingCriterion:
    
    @abstractmethod
    def should_stop(training_error_history, validation_error_history) -> bool:
        pass