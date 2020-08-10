class AgeStructuredModel:
    """
    A class which contains all necessary methods for analyzing an age-structured model for penguin colonies. Objects are
    initialized by the number of assumed adult stages.
    """
    def __init__(self, J=5):
        self.num_stages = J
