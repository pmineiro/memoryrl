import math

from abc import ABC, abstractmethod
from typing import Sequence

from coba.random import CobaRandom
from coba.learners import UcbBanditLearner

import torch
import torch.nn
import torch.optim

from vowpalwabbit import pyvw
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

    def __init__(self, base="none"):

        assert base in ["none", "l1", "l2", "l2^2", "cos", "exp"]

        self.base = base

    def calculate_base(self, query_context, mem_context):
        if self.base == "none":
            return 0

        x1 = query_context.features
        x2 = mem_context.features

        if self.base == "cos":
            n1 = self._norm(x1,"l2")
            n2 = self._norm(x2,"l2")
            return (1-self._dot(x1,x2)/(n1*n2))/2

        if self.base == "exp":
            return math.exp(-self._metric(x1,x2,"l2"))

        return self._metric(x1,x2,self.base)

    def _norm(self,x1,d):
        if isinstance(x1,dict):
            vs = x1.values()
        else:
            vs = x1
        
        if d=="l1":
            return sum(abs(v) for v in vs)
        
        if d=="l2":
            return math.sqrt(sum((v)**2 for v in vs))

        if d=="l2^2":
            return sum((v)**2 for v in vs)

    def _metric(self,x1,x2,d):
        if isinstance(x1,dict) and isinstance(x2,dict):
            vs = (x1.get(k,0)-x2.get(k,0) for k in (x1.keys() | x2.keys()))
        else:
            vs = (i-j for i,j in zip(x1,x2))
        
        self._norm(vs,d)

    def _dot(self,x1,x2):
        if isinstance(x1,dict) and isinstance(x2,dict):
            return sum([x1[k]*x2[k] for k in (x1.keys() & x2.keys())])
        else:
            return sum([i*j for i,j in zip(x1,x2)])

    def __repr__(self) -> str:
        return f"base({self.base})"

    def __str__(self) -> str:
        return self.__repr__()

class TorchScorer(Scorer):

    def __init__(self, l2=0, power_t=0, base:BaseMetric=BaseMetric(), example:Example=InteractionExample()):

        self.example = example
        self.base = base
        self.t = 0
        self.model = None

    def predict(self, query_key, memory_keys):

        if self.model == None:

            class LinearRegression(torch.nn.Module):
                def __init__(s, inputSize, outputSize):
                    super().__init__()
                    s.weights = torch.nn.Parameter(torch.zeros((inputSize, outputSize)))
                    s.linear = torch.nn.Linear(inputSize, outputSize,bias=False)

                def forward(s, x):
                    base     = self.base.calculate_base(*x)
                    features = torch.tensor([ f[1] for f in self.example.feat(*x)])

                    return base+ (features @ s.weights.exp())[0]

            self.model     = LinearRegression(len(query_key.features), 1)
            self.criterion = torch.nn.MSELoss() 
            self.optimizer = torch.optim.Adam(self.model.parameters())

        values = []

        with torch.no_grad():
            for memory_key in memory_keys:
                values.append(-self.model((query_key,memory_key)).item())

        return values

    def update(self, query_key, memory_keys, r):

        self.t += 1

        if len(memory_keys) == 0: return

        self.optimizer.zero_grad()

        if len(memory_keys) >= 1:
            # get output from the model, given the inputs
            outputs = self.model((query_key, memory_keys[0]))
            # get loss for the predicted output
            loss = self.criterion(outputs, torch.tensor(1-r).float())
            # get gradients w.r.t to parameters
            loss.backward()

        if len(memory_keys) >= 2:
            # get output from the model, given the inputs
            outputs = self.model((query_key, memory_keys[1]))
            # get loss for the predicted output
            loss = self.criterion(outputs, torch.tensor(r).float())
            # get gradients w.r.t to parameters
            loss.backward()

        self.optimizer.step()

    def __repr__(self) -> str:
        return f"torch"

    def __str__(self) -> str:
        return self.__repr__()

class RegrScorer(Scorer):

    def __init__(self, l2=0, power_t=0, base:BaseMetric=BaseMetric(), example:Example=InteractionExample()):

        self.base = base

        vw_ignore_linear = " ".join([ f"--ignore_linear {i}" for i in example.ignored() ])
        vw_interactions  = " ".join([ f"--interactions {i}" for i in example.interactions() ])

        self.example = example

        self.vw = pyvw.vw(f'--quiet -b {bits} --l2 {l2} --power_t {power_t} --noconstant --min_prediction 0 --max_prediction 2 {vw_ignore_linear} {vw_interactions}')

        self.t       = 0
        self.l2      = l2
        self.power_t = power_t

        self.rng = CobaRandom(1)

        self.args = (l2, power_t, base, example)

    def __reduce__(self):
        return (type(self), self.args)

    def predict(self, query_key, memory_keys):

        values = []

        for memory_key in memory_keys:

            base    = self.base.calculate_base(query_key, memory_key)
            example = self.example.make_example(self.vw, query_key, memory_key, base)
            values.append(-self.vw.predict(example))
            self.vw.finish_example(example)

        return values

    def update(self, query_key, memory_keys, r):

        self.t += 1

        if len(memory_keys) >= 1:
            base    = self.base.calculate_base(query_key, memory_keys[0]) # 0->1 where 1 is bad
            example = self.example.make_example(self.vw, query_key, memory_keys[0], base, 1-r, 1) #0
            self.vw.learn(example)
            self.vw.finish_example(example)

        if len(memory_keys) >= 2:
            base    = self.base.calculate_base(query_key, memory_keys[1]) 
            example = self.example.make_example(self.vw, query_key, memory_keys[1], base, 1, 1) #1
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
