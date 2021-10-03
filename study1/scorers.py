from coba.random import CobaRandom
from coba.learners import UcbBanditLearner
from vowpalwabbit import pyvw

from scipy.spatial import distance

import numpy as np
import scipy.sparse as sp

from examples import MemExample, InteractionExample, DifferenceExample

bits = 20

class Base:

    def __init__(self, base="none", maxnorm=False):

        assert base in ["none", "mem", "l1", "l2", "l2^2", "cos"]

        self.base    = base
        self.maxnorm = maxnorm
        self.max     = -np.inf

    def calculate_base(self, query_context, mem_context, mem_value):
        base = self._calculate_base(query_context, mem_context, mem_value)
        self.max = max(self.max,base)

        if self.maxnorm:
            return base/(self.max+int(self.max==0))
        else:
            return base

    def _calculate_base(self, query_context, mem_context, mem_value):
        if self.base == "none":
            return 0

        if self.base == "mem":
            return mem_value

        if self.base == "l1":
            ef1 = query_context.features()
            ef2 = mem_context.features()

            if sp.issparse(ef1):
                data = (ef1-ef2).data
                return 0 if len(data) == 0 else distance.minkowski((ef1-ef2).data,0,p=1)
            else:
                return distance.minkowski(ef1,ef2,p=1)

        if self.base == "l2":
            ef1 = query_context.features()
            ef2 = mem_context.features()

            if sp.issparse(ef1):
                data = (ef1-ef2).data
                return 0 if len(data) == 0 else distance.euclidean(data,0)
            else:
                return distance.euclidean(ef1,ef2)

        if self.base == "l2^2":
            ef1 = query_context.features()
            ef2 = mem_context.features()

            if sp.issparse(ef1):
                data = (ef1-ef2).data
                return 0 if len(data) == 0 else distance.sqeuclidean((ef1-ef2).data,0)
            else:
                return distance.sqeuclidean(ef1,ef2)

        if self.base == "cos":
            ef1 = query_context.features()
            ef2 = mem_context.features()

            if sp.issparse(ef1):
                n1 = distance.euclidean(ef1.data,0)
                n2 = distance.euclidean(ef2.data,0)
                return (1-(ef1.T @ ef2)[0,0]/(n1*n2))/2
            else:
                return distance.cosine(ef1, ef2) / 2

    def __repr__(self) -> str:
        return f"base({self.base})"

    def __str__(self) -> str:
        return self.__repr__()

class RegressionScorer:

    def __init__(self, l2=0, power_t=0, baser:Base=Base(), exampler:MemExample=InteractionExample()):

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

            base    = 1-self.baser.calculate_base(xraw, z[0], z[1])
            example = self.exampler.make_example(self.vw, xraw, z[0], base)
            values.append(self.vw.predict(example))
            self.vw.finish_example(example)

        return values

    def update(self, xraw, zs, r):

        self.t += 1
        base    = 1-self.baser.calculate_base(xraw, zs[0][0], zs[0][1])
        example = self.exampler.make_example(self.vw, xraw, zs[0][0], base, r)
        self.vw.learn(example)
        self.vw.finish_example(example)

    def __repr__(self) -> str:
        return f"regr{self.args}"

    def __str__(self) -> str:
        return self.__repr__()

class RankScorer:

    def __init__(self, l2=0, power_t=0, baser:Base=Base(), exampler:MemExample=DifferenceExample()):

        self.baser = baser

        vw_ignore_linear = " ".join([ f"--ignore_linear {i}" for i in exampler.ignored() ])
        vw_interactions  = " ".join([ f"--interactions {i}"  for i in exampler.interactions() ])

        self.vw = pyvw.vw(f'--quiet -b {bits} --l2 {l2} --power_t {power_t} --noconstant --loss_function logistic --link=glf1 {vw_ignore_linear} {vw_interactions}')

        self.t        = 0
        self.l2       = l2
        self.power_t  = power_t
        self.exampler = exampler

        self.rng = CobaRandom(1)

        self.args = (l2, power_t, baser, exampler)

    def __reduce__(self):
        return (type(self), self.args)

    def predict(self, xraw, zs):

        values = []

        for z in zs:
            base    = -self.baser.calculate_base(xraw, z[0], z[1]) 
            example = self.exampler.make_example(self.vw, xraw, z[0], base)
            values.append(self.vw.predict(example))
            self.vw.finish_example(example)

        return values

    def update(self, xraw, zs, r):

        self.t += 1

        if len(zs) >= 1:
            base    = -self.baser.calculate_base(xraw, zs[0][0], zs[0][1]) 
            example = self.exampler.make_example(self.vw, xraw, zs[0][0], base, 1, r)
            self.vw.learn(example)
            self.vw.finish_example(example)

        if len(zs) >= 2:
            base    = -self.baser.calculate_base(xraw, zs[1][0], zs[1][1]) 
            example = self.exampler.make_example(self.vw, xraw, zs[1][0], base, -1, r)
            self.vw.learn(example)
            self.vw.finish_example(example)

    def __repr__(self) -> str:
        return f"rank{self.args}"

    def __str__(self) -> str:
        return self.__repr__()


class UCBScorer:
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
