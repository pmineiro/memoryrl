
from abc import ABC, abstractmethod
from typing import Sequence

from vowpalwabbit import pyvw

from coba.learners import VowpalMediator
from coba.random import CobaRandom

import scipy.sparse as sp
import numpy as np

from sklearn.decomposition import TruncatedSVD

bits = 20

class Router(ABC):

    @abstractmethod
    def predict(self, query_key) -> int:
        ...

    @abstractmethod
    def update(self, query_key, label, weight) -> None:
        ...

class RouterFactory(ABC):

    @abstractmethod
    def create(self, query_keys) -> Router:
        ...

class ProjRouter(RouterFactory):

    class _Router(Router):

        def __init__(self, features, projector, boundary):
            self._features  = features
            self._projector = projector
            self._boundary  = boundary

        def predict(self, query_key):
            value = (query_key.mat(self._features) @ self._projector)[0] - self._boundary
            return np.sign(value)*(1-np.exp(-abs(value)))

        def update(self, query_key, label, weight):
            pass

    def __init__(self, features = ['x'], proj="PCA", samples=50) -> None:
        self.features = tuple(features)
        self.proj     = proj
        self.samples  = samples

    def create(self, keys2split) -> _Router:

        is_sparse = isinstance(keys2split[0].raw(self.features), dict)

        if not is_sparse and len(keys2split[0].raw(self.features)) == 1:
            projector = np.array([1])
            boundary  = np.median([key.raw(self.features)[0] for key in keys2split])            

        elif self.proj == "PCA":
            if is_sparse:
                features = sp.vstack([k.mat(self.features) for k in keys2split])
                center   = sp.vstack([sp.csr_matrix(features.mean(axis=0))]*len(keys2split))
            else:
                features = np.vstack([k.mat(self.features) for k in keys2split])
                center   = np.vstack([features.mean(axis=0)]*len(keys2split))

            projector = TruncatedSVD(n_components=1).fit(features-center).components_.astype(float)[0]
            boundary  = np.median(features @ projector)

        else:
            max_projector   = None
            max_dispersion  = 0
            max_projections = None

            raws2split = [k.raw(self.features) for k in keys2split]

            if is_sparse:
                mat2split  = sp.vstack([k.mat(self.features) for k in keys2split])
            else:
                mat2split  = np.vstack([k.mat(self.features) for k in keys2split])

            if is_sparse:
                indices = list(set(sp.find(mat2split)[1]))
            else:
                indices = list(range(len(raws2split[0])))

            for _ in range(self.samples):
                projector = np.random.randn(len(indices))
                projector = projector/np.linalg.norm(projector)

                if is_sparse:
                    sparse_projector = np.zeros((mat2split.shape[1]),float)
                    sparse_projector[indices] = projector
                    projector = sparse_projector

                projections = projector @ mat2split.T
                dispersion  = projections.var()

                if dispersion > max_dispersion:
                    max_projector   = projector
                    max_dispersion  = dispersion
                    max_projections = projections

            projector = max_projector.T
            boundary  = np.median(max_projections)

        return ProjRouter._Router(self.features, projector, boundary)

    def __str__(self) -> str:

        return f"P{(self.proj,self.features,self.samples)}"

class VowpRouter(RouterFactory):

    class _Router(Router):

        def __init__(self, X: Sequence[str], base: Router):

            options = [
                "--quiet",
                f"-b {bits}",
                f"--power_t {0}",
                f"--random_seed {1}",
                "--coin",
                "--noconstant",
                "--loss_function logistic",
                "--link=glf1",
            ]

            X = X or ['x','a']
            if 'x' not in X: options.append("--ignore_linear x")
            if 'a' not in X: options.append("--ignore_linear a")
            options.extend([f"--interactions {x}" for x in X if len(x) > 1])

            self.vw = VowpalMediator().init_learner(" ".join(options), 1)
            self.base = base

        def predict(self, query_key):
            return self.vw.predict(self._make_example(query_key, None, None))

        def update(self, query_key, label, weight):
            self.vw.learn(self._make_example(query_key, label, weight))

        def _make_example(self, query_key, label, weight) -> pyvw.example:
            label = f"{0 if label is None else label} {0 if weight is None else weight} {self.base.predict(query_key)}"
            return self.vw.make_example({"x": query_key.raw('x'), "a": query_key.raw('a') }, label)

    def __init__(self, X:Sequence[str]=[], base="PCA") -> None:
        self._args = (tuple(X),base)

    def create(self, keys2split) -> _Router:

        new_router = VowpRouter._Router(self._args[0], ProjRouter(proj=self._args[1]).create(keys2split))

        return new_router

    def __str__(self) -> str:
        return f"vw{self._args}"

class RandomRouter(RouterFactory,Router):

    def __init__(self,*args,**kwargs) -> None:
        self._rng = CobaRandom(1)

    def create(self, keys2split) -> 'RandomRouter':
        return self

    def predict(self, xraw):
        return self._rng.random()-0.5

    def update(self, xraw, y, w):
        pass

    def __repr__(self) -> str:
        return f"rand"

    def __str__(self) -> str:
        return self.__repr__()

