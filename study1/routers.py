from abc import ABC, abstractmethod
from typing import Sequence

from vowpalwabbit import pyvw
from coba.learners.vowpal import VowpalMediator
from coba.random import CobaRandom

import scipy.sparse as sp
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA

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

class PCARouter(RouterFactory):
    class _Router(Router):

        def __init__(self, median, feature_avg, first_component):
            self.median = median
            self.feature_avg = feature_avg
            self.first_component = first_component

        def predict(self, query_key):
            value = (self.median - self.first_component @ (query_key.features - self.feature_avg).T).item()
            return np.sign(value)*(1-np.exp(-abs(value)))

        def update(self, query_key, label, weight):
            pass

    def __init__(self, *args, **kwargs) -> None:
        pass

    def create(self, split_keys) -> _Router:
        if sp.issparse(split_keys[0].features):
            features_mat  = sp.vstack([k.features for k in split_keys])
            features_avg  = features_mat.mean(axis=0)
            features_mat -= sp.vstack([sp.csr_matrix(features_avg)]*len(split_keys))
            
        else:
            features_mat  = np.vstack([k.features for k in split_keys])
            features_avg  = features_mat.mean(axis=0)
            features_mat -= np.vstack([features_avg]*len(split_keys))
            

        first_component   = TruncatedSVD(n_components=1).fit(features_mat).components_
        first_projections = (first_component @ features_mat.T).squeeze().tolist()

        return PCARouter._Router(np.median(first_projections), features_avg, first_component)

    def __str__(self) -> str:
        return f"PCA"

class LogisticRouter(RouterFactory):

    class _Router(Router):

        def __init__(self, power_t, X, coin, l2, seed, base):
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
            self.base = base

        def predict(self, query_key):
            return self.vw.predict(self._make_example(query_key, None, None))

        def update(self, query_key, label, weight):
            self.vw.learn(self._make_example(query_key, label, weight))

        def _make_example(self, query_key, label, weight) -> pyvw.example:
            label = f"{0 if label is None else label} {0 if weight is None else weight} {self.base.predict(query_key)}"
            return self.vw.make_example({"x": query_key.context, "a": query_key.action}, label)

    def __init__(self, power_t:float=0, X:Sequence[str]=[], coin:bool=True, l2: float=0, seed:int=1) -> None:
        self._args = (power_t,X,coin,l2,seed)
        self._rng  = CobaRandom(seed)

    def create(self, split_keys) -> _Router:

        pca_router = PCARouter().create(split_keys)
        new_router = LogisticRouter._Router(*self._args, pca_router)

        return new_router

    def __str__(self) -> str:
        return f"vw{self._args}"

class RandomRouter(RouterFactory,Router):

    def __init__(self, *args, **kwargs) -> None:
        self._rng = CobaRandom(1)

    def create(self, query_keys) -> 'RandomRouter':
        return self

    def predict(self, xraw):
        return self._rng.random()-0.5

    def update(self, xraw, y, w):
        pass

    def __repr__(self) -> str:
        return f"rand"

    def __str__(self) -> str:
        return self.__repr__()
