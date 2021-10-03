
class RewardFeedback:
    
    def signal(self, observed:float, memory:float, reward:float) -> float:
        return reward

    @property
    def params(self):
        return ('reward',)

class DeviationFeedback:

    def __init__(self, type:str = "squared"):
        assert type in ["squared", "absolute"]
        self._type = type

    @property
    def params(self):
        return ('deviation', self._type)
        
    def signal(self, observed:float, memory:float, reward:float) -> float:

        deviation = observed-memory

        if self._type == "squared":
            return 1-deviation**2
        
        if self._type == "absolute":
            return 1-abs(deviation)
