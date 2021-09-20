from coba.random import CobaRandom
from coba.learners import UcbBanditLearner
from vowpalwabbit import pyvw

from scipy.sparse import issparse
from scipy.spatial import distance
import numpy as np

from examplers import Exampler, PureExampler, DiffSquareExampler

bits = 20

class Baser:

    def __init__(self, base="none"):
        
        assert base in ["none", "mem", "l2", "cos"]

        self.base = base

    def calculate_base(self, query_context, mem_context, mem_value):
        if self.base == "none":
            return 0

        if self.base == "mem":
            return mem_value

        if self.base == "l2":
            ef1 = query_context.features()
            ef2 = mem_context.features()

            if issparse(ef1):
                return distance.sqeuclidean((ef1-ef2).data,0) / ef1.shape[1]
            else:
                return distance.sqeuclidean(ef1,ef2) / ef1.shape[1]

        if self.base == "cos":
            ef1 = query_context.features()
            ef2 = mem_context.features()

            if issparse(ef1):
                n1 = distance.euclidean(ef1.data,0)
                n2 = distance.euclidean(ef2.data,0)
                return (1-(ef1.T @ ef2)[0,0]/(n1*n2))/2
            else:
                return distance.cosine(ef1, ef2) / 2

class RegrScorer:

    def __init__(self, l2=0, power_t=0, base="none", exampler:Exampler=PureExampler(), interactions=["ac","ad","bc","bd","abcd"], ignored=["a","b","c","d"]):

        self.baser = Baser(base)

        vw_ignore_linear = " ".join([ f"--ignore_linear {i}" for i in ignored ])
        vw_interactions  = " ".join([ f"--interactions {i}" for i in interactions ])

        self.exampler = exampler

        self.vw = pyvw.vw(f'--quiet -b {bits} --l2 {l2} --power_t {power_t} --min_prediction -1 --max_prediction 2 {vw_ignore_linear} {vw_interactions}')

        self.t       = 0
        self.l2      = l2
        self.power_t = power_t
        self.base    = base

        self.rng = CobaRandom(1)

        self.args = (l2, power_t, base, exampler, interactions, ignored)

    def __reduce__(self):
        return (type(self), self.args)

    @property
    def params(self):
        return ('regr',) + self.args

    def predict(self, xraw, zs):

        values = []

        for z in zs:
            base    = self.baser.calculate_base(xraw, z[0], z[1])
            example = self.exampler.make_example(self.vw, xraw, z[0], base)
            values.append(self.vw.predict(example))

        return values

    def update(self, xraw, zs, r):

        self.t += 1
        base    = self.baser.calculate_base(xraw, zs[0][0], zs[0][1])
        example = self.exampler.make_example(self.vw, xraw, zs[0][0], base, r)
        self.vw.learn(example)

class ClassScorer:

    def __init__(self, l2=0, power_t=0, base="none", exampler:Exampler=DiffSquareExampler(), interactions=[], ignored=[]):

        self.baser = Baser(base)

        vw_ignore_linear = " ".join([ f"--ignore_linear {i}" for i in ignored ])
        vw_interactions  = " ".join([ f"--interactions {i}" for i in interactions ])

        self.vw = pyvw.vw(f'--quiet -b {bits} --l2 {l2} --power_t {power_t} --noconstant --loss_function logistic --link=glf1 {vw_ignore_linear} {vw_interactions}')

        self.t        = 0
        self.l2       = l2
        self.power_t  = power_t
        self.exampler = exampler

        self.rng = CobaRandom(1)

        self.args = (l2, power_t, base, exampler, interactions, ignored)

    def __reduce__(self):
        return (type(self), self.args)

    @property
    def params(self):
        return ('class',) + self.args

    def predict(self, xraw, zs):

        values = []

        for z in zs:
            base    = self.baser.calculate_base(xraw, z[0], z[1]) 
            example = self.exampler.make_example(self.vw, xraw, z[0], base)
            values.append(self.vw.predict(example))

        return values

    def update(self, xraw, zs, r):

        self.t += 1

        weight = r

        if len(zs) >= 1:
            base    = self.baser.calculate_base(xraw, zs[0][0], zs[0][1]) 
            example = self.exampler.make_example(self.vw, xraw, zs[0][0], base, 1, weight)
            self.vw.learn(example)

        if len(zs) >= 2:
            base    = self.baser.calculate_base(xraw, zs[1][0], zs[1][1]) 
            example = self.exampler.make_example(self.vw, xraw, zs[1][0], base, -1, weight)
            self.vw.learn(example)

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

class DistanceScorer:

    def __init__(self, order=2, norm="mean"):

        assert norm in ['max','mean']

        self.i     = 0
        self.stat  = norm
        self.order = order
        self.norm  = 1

    @property
    def params(self):
        return ('distance', self.order, self.stat)

    def distance(self,x1,x2):
        diff = x1.features()-x2.features()

        if  issparse(diff):
            return np.linalg.norm(diff.data, ord=self.order)**2
        else:
            return np.linalg.norm(diff     , ord=self.order)**2

    def predict(self, xraw, zs):
        values = []

        for z in zs:
            values.append(-self.distance(xraw, z[0]) / (self.norm if self.norm != 0 else 1))

        return values

    def update(self, xraw, zs, r):

        for z in zs:
            self.i += 1
            new_observation = self.distance(xraw, z[0])

            if self.stat == 'max':
                self.norm = max(self.norm, new_observation)

            if self.stat == 'mean':
                self.norm = (1-1/self.i) * self.norm + (1/self.i) * new_observation

class AdditionScorer:

    def __init__(self, scorers):
        self.scorers = scorers

    def predict(self, xraw, zs):
        return list(map(sum,zip(*[scorer.predict(xraw,zs) for scorer in self.scorers])))

    def update(self, xraw, zs, r):
        for scorer in self.scorers:
            scorer.update(xraw, zs, r)

    @property
    def params(self):
        return ('addition', [ scorer.params for scorer in self.scorers])