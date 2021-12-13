from abc import ABC, abstractmethod
from typing import Sequence

from coba.random import CobaRandom
from coba.learners import UcbBanditLearner
from vowpalwabbit import pyvw

from scipy.spatial import distance

import numpy as np
import scipy.sparse as sp

from examples import Example, InteractionExample, DiffExample

bits = 20

class Scorer(ABC):

    @abstractmethod
    def predict(self, query_key, memory_keys) -> Sequence[float]:
        ...

    @abstractmethod
    def update(self, query_key, memory_keys, y):
        ...

class BaseMetric:

    def __init__(self, base="none", maxnorm=False):

        assert base in ["none", "l1", "l2", "l2^2", "cos", "exp"]

        self.base    = base
        self.maxnorm = maxnorm
        self.max     = -np.inf

    def calculate_base(self, query_context, mem_context):
        base = self._calculate_base(query_context, mem_context)
        self.max = max(self.max,base)

        if self.maxnorm:
            return base/(self.max+int(self.max==0))
        else:
            return base

    def _calculate_base(self, query_context, mem_context):
        if self.base == "none":
            return 0

        ef1 = query_context.features()
        ef2 = mem_context.features()

        if self.base == "l1":
            if sp.issparse(ef1):
                data = (ef1-ef2).data
                return 0 if len(data) == 0 else distance.minkowski((ef1-ef2).data,0,p=1)
            else:
                return distance.minkowski(ef1,ef2,p=1)

        if self.base == "l2":
            if sp.issparse(ef1):
                data = (ef1-ef2).data
                return 0 if len(data) == 0 else distance.euclidean(data,0)
            else:
                return distance.euclidean(ef1,ef2)

        if self.base == "l2^2":
            if sp.issparse(ef1):
                data = (ef1-ef2).data
                return 0 if len(data) == 0 else distance.sqeuclidean((ef1-ef2).data,0)
            else:
                return distance.sqeuclidean(ef1,ef2)

        if self.base == "cos":
            if sp.issparse(ef1):
                n1 = distance.euclidean(ef1.data,0)
                n2 = distance.euclidean(ef2.data,0)
                return (1-self._sparse_dp(ef1,ef2)/(n1*n2))/2
            else:
                return distance.cosine(ef1, ef2) / 2

        if self.base == "exp":
            if sp.issparse(ef1):
                data = (ef1-ef2).data
                return 0 if len(data) == 0 else np.exp(-distance.euclidean(data,0))
            else:
                return 1-np.exp(-distance.euclidean(ef1,ef2))

    def _sparse_dp(self,x1,x2):

        x1.sort_indices()
        x2.sort_indices()

        idx1 = x1.indices
        idx2 = x2.indices

        dat1 = x1.data
        dat2 = x2.data

        i=0
        j=0

        v = 0

        while True:

            if i == len(idx1) or j == len(idx2):
                return v

            idx1_i = idx1[i]
            idx2_j = idx2[j]

            if idx1_i == idx2_j:
                v += dat1[i]*dat2[j]
            
            if idx1_i <= idx2_j:
                i += 1

            if idx2_j <= idx1_i:
                j += 1

    def __repr__(self) -> str:
        return f"base({self.base})"

    def __str__(self) -> str:
        return self.__repr__()

class RegressionScorer(Scorer):

    def __init__(self, l2=0, power_t=0, baser:BaseMetric=BaseMetric(), exampler:Example=InteractionExample()):

        self.baser = baser

        vw_ignore_linear = " ".join([ f"--ignore_linear {i}" for i in exampler.ignored() ])
        vw_interactions  = " ".join([ f"--interactions {i}" for i in exampler.interactions() ])

        self.exampler = exampler

        self.vw = pyvw.vw(f'--quiet -b {bits} --l2 {l2} --power_t {power_t} --min_prediction -1 --max_prediction 2 {vw_ignore_linear} {vw_interactions}')

        self.t       = 0
        self.l2      = l2
        self.power_t = power_t

        self.rng = CobaRandom(1)

        self.args = (l2, power_t, baser, exampler)

    def __reduce__(self):
        return (type(self), self.args)

    def predict(self, xraw, zs):

        values = []

        for z in zs:

            base    = 1-self.baser.calculate_base(xraw, z[0])
            example = self.exampler.make_example(self.vw, xraw, z[0], base)
            values.append(self.vw.predict(example))
            self.vw.finish_example(example)

        return values

    def update(self, xraw, zs, r):

        self.t += 1
        base    = 1-self.baser.calculate_base(xraw, zs[0][0])
        example = self.exampler.make_example(self.vw, xraw, zs[0][0], base, r)
        self.vw.learn(example)
        self.vw.finish_example(example)

    def __repr__(self) -> str:
        return f"regr{self.args}"

    def __str__(self) -> str:
        return self.__repr__()

class RankScorer(Scorer):

    def __init__(self, l2=0, power_t=0, base:BaseMetric=BaseMetric(), example:Example=DiffExample()):

        self.base = base

        vw_ignore_linear = " ".join([ f"--ignore_linear {i}" for i in example.ignored() ])
        vw_interactions  = " ".join([ f"--interactions {i}"  for i in example.interactions() ])

        self.vw = pyvw.vw(f'--quiet -b {bits} --l2 {l2} --power_t {power_t} --noconstant --loss_function logistic --link=glf1 {vw_ignore_linear} {vw_interactions}')

        self.t        = 0
        self.l2       = l2
        self.power_t  = power_t
        self.example = example

        self.rng = CobaRandom(1)

        self.args = (l2, power_t, base, example)

    def __reduce__(self):
        return (type(self), self.args)

    def predict(self, query_key, memory_keys):

        values = []

        for z in memory_keys:
            base    = 1-2*self.base.calculate_base(query_key, z)
            example = self.example.make_example(self.vw, query_key, z, base)
            values.append(self.vw.predict(example))
            self.vw.finish_example(example)

        return values

    def update(self, query_key, memory_keys, r):

        self.t += 1

        if len(memory_keys) >= 1:
            base    = 1-2*self.base.calculate_base(query_key, memory_keys[0])
            example = self.example.make_example(self.vw, query_key, memory_keys[0], base, 1, r) #0
            self.vw.learn(example)
            self.vw.finish_example(example)

        if len(memory_keys) >= 2:
            base    = 1-2*self.base.calculate_base(query_key, memory_keys[1]) 
            example = self.example.make_example(self.vw, query_key, memory_keys[1], base, -1, r) #1
            self.vw.learn(example)
            self.vw.finish_example(example)

    def __repr__(self) -> str:
        return f"rank{self.args}"

    def __str__(self) -> str:
        return self.__repr__()

class UCBScorer(Scorer):
    def __init__(self):
        self._ucb_learner = UcbBanditLearner()

    @property
    def params(self):
        return ('UCB',)

    def reset(self, zs):
        for z in zs:
            if z[0] in self._ucb_learner._m:
                pass#del self._ucb_learner._m[z[0]]

    def predict(self, xraw, zs):

        if zs:
            return self._ucb_learner.predict(None, [z[0] for z in zs])
        else:
            return []

    def update(self, xraw, zs, r):
        self._ucb_learner.learn(None, zs[0][0], r, 1, None)

class RandomScorer(Scorer):

    def __init__(self):
        self.rng = CobaRandom(1)

    def predict(self, xraw, zs):
        return self.rng.randoms(len(zs))

    def update(self, xraw, zs, r):
        pass

    def __repr__(self) -> str:
        return f"rand"

    def __str__(self) -> str:
        return self.__repr__()
