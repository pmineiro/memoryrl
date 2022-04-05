from abc import ABC, abstractmethod
from typing import Sequence

from vowpalwabbit import pyvw
from coba.learners.vowpal import VowpalMediator
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

class PCARouter(RouterFactory):

    class _Router(Router):

        def __init__(self, median, feature_avgs, first_component):
            self.median          = median
            self.feature_avgs    = feature_avgs
            self.first_component = first_component

        def predict(self, query_key):
            value = (self.median - self.first_component @ (query_key.features - self.feature_avgs).T).item()
            return np.sign(value)*(1-np.exp(-abs(value)))

        def update(self, query_key, label, weight):
            pass

    def create(self, keys2split) -> _Router:

        if sp.issparse(keys2split[0].features):
            features_mat  = sp.vstack([k.features for k in keys2split])
            features_avg  = features_mat.mean(axis=0)
            features_mat -= sp.vstack([sp.csr_matrix(features_avg)]*len(keys2split))
        else:
            features_mat  = np.vstack([k.features for k in keys2split])
            features_avg  = features_mat.mean(axis=0)
            features_mat -= np.vstack([features_avg]*len(keys2split))

        first_component   = TruncatedSVD(n_components=1).fit(features_mat).components_
        first_projections = (first_component @ features_mat.T).squeeze().tolist()

        return PCARouter._Router(np.median(first_projections), features_avg, first_component)

    def __str__(self) -> str:
        return f"PCA"

class LogisticRouter(RouterFactory):

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
            return self.vw.make_example({"x": query_key.context, "a": query_key.action}, label)

    def __init__(self, X:Sequence[str]=[]) -> None:
        self._args = (X,)

    def create(self, keys2split) -> _Router:

        pca_router = PCARouter().create(keys2split)
        new_router = LogisticRouter._Router(*self._args, pca_router)

        return new_router

    def __str__(self) -> str:
        return f"vw{self._args}"

class RandomRouter(RouterFactory,Router):

    def __init__(self) -> None:
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
