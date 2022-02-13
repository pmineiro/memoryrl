from math import log
from abc import ABC, abstractmethod

class Splitter(ABC):

    @abstractmethod
    def __call__(self, n:int) -> int:
        ...

class NeverSplitter(Splitter):

    def __call__(self, n: int) -> int:
        return float('inf')
    
    def __str__(self) -> str:
        return f"never"

    def __repr__(self) -> str:
        return self.__str__()

class ConstSplitter(Splitter):

    def __init__(self, const: int) -> None:
        self.const = const

    def __call__(self, n: int) -> int:
        return self.const
    
    def __str__(self) -> str:
        return f"const({self.const})"

    def __repr__(self) -> str:
        return self.__str__()

class LogSplitter(Splitter):
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def __call__(self, n: int) -> int:
        return self.scale * log(n+1)
    
    def __str__(self) -> str:
        return f"log({self.scale})"

    def __repr__(self) -> str:
        return self.__str__()