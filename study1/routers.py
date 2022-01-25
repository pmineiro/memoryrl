from abc import ABC, abstractmethod
from typing import Sequence

from vowpalwabbit import pyvw
from coba.learners.vowpal import VowpalMediator
from coba.random import CobaRandom

bits = 20

class Router(ABC):

    @abstractmethod
    def predict(self, x) -> int:
        ...
    
    @abstractmethod
    def update(self, query_key, label, weight):
        ...

class RouterFactory(ABC):

    @abstractmethod
    def __call__(self) -> Router:
        ...

class LogisticRouter(RouterFactory):

    class _Router(Router):

        def __init__(self, power_t, X, coin, l2):
            self.t  = 0
            options = [
                "--quiet",
                f"-b {bits}",
                f"--power_t {power_t}",
                "--coin" if coin else "",
                "--noconstant",
                f"--l2 {l2}",
                "--loss_function logistic",
                "--link=glf1",
                *[f"--interactions {x}" for x in X]
            ]
            self.vw = VowpalMediator().init_learner(" ".join(options), 0)

        def predict(self, query_key):
            return self.vw.predict(self._make_example(query_key, None, None))

        def update(self, query_key, label, weight):
            self.t += 1
            self.vw.learn(self._make_example(query_key, label, weight))

        def _make_example(self, query_key, label, weight) -> pyvw.example:
            label = None if label is None else f"{label} {weight}"
            return self.vw.make_example({"x": query_key.context, "a": query_key.action}, label)

    def __init__(self, power_t:float, X:Sequence[str], coin:bool, l2: float) -> None:
        self._args = (power_t, X, coin, l2)

    def __call__(self) -> _Router:
        return LogisticRouter._Router(*self._args)

    def __str__(self) -> str:
        return f"vw{self._args}"

class RandomRouter(RouterFactory,Router):

    def __init__(self) -> None:
        self._rng = CobaRandom(1)

    def __call__(self) -> 'RandomRouter':
        return self

    def predict(self, xraw):
        return 1-2*self._rng.random()

    def update(self, xraw, y, w):
        pass

    def __repr__(self) -> str:
        return f"rand"

    def __str__(self) -> str:
        return self.__repr__()
