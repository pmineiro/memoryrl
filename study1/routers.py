import random
from abc import ABC, abstractmethod

from vowpalwabbit import pyvw
from examples import IdentityExample

bits = 20

class Router(ABC):

    @abstractmethod
    def predict(self, x) -> int:
        ...
    
    @abstractmethod
    def update(self, x, y, weight):
        ...

class RouterFactory(ABC):

    @abstractmethod
    def __call__(self) -> Router:
        ...

class Logistic_VW(RouterFactory):

    class Logistic_VW_Router(Router):

        def __init__(self, vw, index):
            self.vw = vw
            self.example = IdentityExample(int(index*2**bits))
            self.t = 0

        def predict(self, xraw):
            return self.vw.predict(self.example.make_example(self.vw, xraw.features))

        def update(self, xraw, y, w):
            self.t+=1
            self.vw.learn(self.example.make_example(self.vw, xraw.features, 0, y, w))

    def __init__(self, power_t:float=0) -> None:
        # We add 10 to bits which means we can represent
        # 2**10 internal nodes for each vw instance that we create
        # VW expects a 32 bit unsinged integer so our hash value plus the offset must be < 2**32
        # This means we have to create multiple VW instances to make sure we stay below this
        self._vws     = [] 
        self._index   = -1
        self._power_t = power_t

    def __call__(self) -> Logistic_VW_Router:
        self._index += 1

        if self._index % 1000 == 0:
            self._vws.append(pyvw.vw(f'--quiet -b {bits+10} --loss_function logistic --noconstant --power_t {self._power_t} --link=glf1 --sparse_weights'))

        vw    = self._vws[int(self._index/1000)]
        index = self._index % 1000

        return Logistic_VW.Logistic_VW_Router(vw, index)

    def __repr__(self) -> str:
        return f"vw(power_t={self._power_t})"

    def __str__(self) -> str:
        return self.__repr__()

    def __reduce__(self):
        return (type(self),(self._power_t,))

class Logistic_SK(RouterFactory):

    class Logistic_SK_Router(Router):
        def __init__(self):

            from sklearn.linear_model import SGDClassifier
            self.clf  = SGDClassifier(loss="log", average=True, learning_rate='constant', eta0=0.5)
            self.is_fit = False

        def predict(self, x):
            return 1 if not self.is_fit else self.clf.predict(self._domain(x))[0]

        def update(self, x, y, w):
            self.clf.partial_fit(self._domain(x), [y], sample_weight=[w], classes=[-1,1])
            self.is_fit = True

        def _domain(self, x):
            return x.features()
    
    def __call__(self) -> Logistic_SK_Router:
        return Logistic_SK.Logistic_SK_Router()

    def __repr__(self) -> str:
        return f"sk"
    
    def __str__(self) -> str:
        return self.__repr__()

class RandomRouter(RouterFactory,Router):

    def __init__(self) -> None:
        self._rng = random.Random(1)

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
