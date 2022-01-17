from abc import ABC, abstractmethod
from typing import Sequence

from vowpalwabbit import pyvw
from coba.learners.vowpal import VowpalMediator

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

        def __init__(self, power_t, X, coin):
            self.t  = 0
            interactions = " ".join(f"--interactions {x}" for x in X)
            coin_flag    = "--coin" if coin else ""
            self.vw = pyvw.vw(f'--quiet -b {bits} --loss_function logistic {coin_flag} --noconstant --power_t {power_t} --link=glf1 {interactions}')

        def predict(self, query_key):
            example = self._make_example(query_key, None, None)
            value = self.vw.predict(example)
            self.vw.finish_example(example)
            
            return value

        def update(self, query_key, label, weight):
            self.t += 1

            example = self._make_example(query_key, label, weight)
            self.vw.learn(example)
            self.vw.finish_example(example)

        def _make_example(self, query_key, label, weight) -> pyvw.example:

            context = query_key.context
            action  = query_key.action

            x = VowpalMediator.prep_features(context)
            a = VowpalMediator.prep_features(action)

            ex = pyvw.example(self.vw, {"x": x, "a": a})

            if label is not None:
                ex.set_label_string(f"{label} {weight}")

            return ex

    def __init__(self, power_t:float, X:Sequence[str], coin:bool) -> None:
        self._power_t = power_t
        self._X = X
        self._coin = coin

    def __call__(self) -> _Router:
        return LogisticRouter._Router(self._power_t, self._X, self._coin)

    def __str__(self) -> str:
        return f"vw({self._power_t},{self._X},{self._coin})"
