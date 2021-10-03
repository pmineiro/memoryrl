
class RewardFeedback:
    
    def signal(self, observed:float, memory:float, reward:float) -> float:
        return reward

    def __repr__(self) -> str:
        return f"reward"

    def __str__(self) -> str:
        return self.__repr__()



class DeviationFeedback:

    def __init__(self, type:str = "^2"):
        assert type in ["^2", "abs"]
        self._type = type
        
    def signal(self, observed:float, memory:float, reward:float) -> float:

        deviation = observed-memory

        if self._type == "^2":
            return 1-deviation**2
        
        if self._type == "abs":
            return 1-abs(deviation)

    def __repr__(self) -> str:
        return f"dev({self._type})"

    def __str__(self) -> str:
        return self.__repr__()


