class SequenceTooLongError(ValueError):
    """ Raised when a sequence exceeds what the model can process in a single forward context.

    Deliberately narrow: callers that can skip a single sequence and carry on catch this and nothing else.
    """
