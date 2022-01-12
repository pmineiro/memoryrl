import math

from abc import ABC, abstractmethod
from typing import Sequence

from coba.random import CobaRandom
from coba.learners import UcbBanditLearner

import torch
import torch.nn
import torch.optim
import torch.sparse

from vowpalwabbit import pyvw
from examples import Example, InteractionExample, DiffExample

bits = 20

class Scorer(ABC):

    @abstractmethod
    def predict(self, query_key, memories) -> Sequence[float]:
        ...

    @abstractmethod
    def update(self, query_key, memories, memory_rwds, prob):
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
            return 1-math.exp(-self._metric(x1,x2,"l1"))

        return self._metric(x1,x2,self.base)

    def _norm(self,vs,d):        
        if d=="l1":
            return sum(abs(v) for v in vs)
        
        if d=="l2":
            return math.sqrt(sum((v)**2 for v in vs))

        if d=="l2^2":
            return sum((v)**2 for v in vs)

    def _metric(self,x1,x2,d):
        
        if isinstance(x1,dict) and isinstance(x2,dict):
            vs = [x1.get(k,0)-x2.get(k,0) for k in (x1.keys() | x2.keys())]
        else:
            vs = [i-j for i,j in zip(x1,x2)]

        return self._norm(vs,d)

    def _dot(self,x1,x2):
        if isinstance(x1,dict) and isinstance(x2,dict):
            return sum([x1[k]*x2[k] for k in (x1.keys() & x2.keys())])
        else:
            return sum([i*j for i,j in zip(x1,x2)])

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"base({self.base})"

class TorchScorer(Scorer):

    def __init__(self, l2=0, power_t=0, v=1, exp=True, optim=True, base:BaseMetric=BaseMetric(), example:Example=InteractionExample()):

        self.example = example
        self.base = base
        self.t = 0
        self.model = None
        self.optim = optim
        self.exp = exp
        self.v = v

    def predict(self, query_key, memories):

        memory_keys = [ mem[0] for mem in memories ]

        if self.model == None:

            is_dense = not isinstance(query_key.context,dict)

            class LinearRegression(torch.nn.Module):
                def __init__(s):
                    super().__init__()

                    if is_dense:
                        s.weights = torch.nn.Parameter(torch.zeros((len(query_key.features),1)), requires_grad=True)
                    else:
                        s.weights = torch.nn.Parameter(torch.zeros((2**bits,1)), requires_grad=True)

                    if self.v in [1,3]:
                        s.transform = torch.nn.Identity()
                    else:
                        if is_dense:
                            s.transform = torch.nn.Softmax(dim=0)
                        else:
                            raise NotImplementedError()

                def forward(s, x):
                    base = self.base.calculate_base(*x)-1

                    if is_dense:
                        features = torch.tensor([[f[1] for f in self.example.feat(*x)]])
                    else:
                        cols,values = zip(*self.example.feat(*x))
                        rows = [0]*len(cols) 
                        features = torch.sparse_coo_tensor([rows,cols],values,(1,2**bits))

                    if is_dense:
                        return base + torch.mm(features,s.transform(s.weights)).squeeze()
                    else:
                        return base + torch.sparse.mm(features,s.transform(s.weights)).squeeze()

            self.model     = LinearRegression()
            self.criterion = torch.nn.MSELoss()

            #if is_dense:
            self.optimizer = torch.optim.Adam(list(self.model.parameters()))
            #else:
            #    self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()))

        values = []

        with torch.no_grad():
            for memory_key in memory_keys:
                values.append(-self.model((query_key,memory_key)).item())

        return values

    def update(self, query_key, memories, memory_rwds, prob):

        memory_keys = [ mem[0] for mem in memories ]
        memory_keys,memory_rwds = zip(*sorted(zip(memory_keys,memory_rwds), key= lambda t: t[1], reverse=True))
        
        self.t += 1

        if len(memory_keys) == 0: return
        if memory_rwds[0] == 0: return

        loss = []

        if len(memory_keys) >= 1:
            # get output from the model, given the inputs
            output = self.model((query_key, memory_keys[0]))
            # get loss for the predicted output
            loss = self.criterion(output, torch.tensor(0).float())

        if len(memory_keys) >= 2:
            # get output from the model, given the inputs
            output = self.model((query_key, memory_keys[1]))
            # get loss for the predicted output
            loss += self.criterion(output, torch.tensor(1).float())

        if self.optim:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.v == 1:
                self.model.weights.data.clamp_min_(0)

    def __repr__(self) -> str:
        return f"torch({self.optim},{self.v},{self.base})"

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

    def predict(self, query_key, memories):

        memory_keys = [m[0] for m in memories]
        values      = []

        for memory_key in memory_keys:

            base    = self.base.calculate_base(query_key, memory_key)
            example = self.example.make_example(self.vw, query_key, memory_key, base)
            values.append(-self.vw.predict(example))
            self.vw.finish_example(example)

        return values

    def update(self, query_key, memories, memory_rwds, prob):

        memory_keys = [ mem[0] for mem in memories ]
        memory_keys,memory_rwds = zip(*sorted(zip(memory_keys,memory_rwds), key= lambda t: t[1], reverse=True))

        self.t += 1

        if len(memory_keys) >= 1:
            base    = self.base.calculate_base(query_key, memory_keys[0]) # 0->1 where 1 is bad
            example = self.example.make_example(self.vw, query_key, memory_keys[0], base, 0, 1) #0
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

    def predict(self, query_key, memories):

        memory_keys = [m[0] for m in memories]

        values = []

        for z in memory_keys:
            base    = 1-2*self.base.calculate_base(query_key, z)
            example = self.example.make_example(self.vw, query_key, z, base)
            values.append(self.vw.predict(example))
            self.vw.finish_example(example)

        return values

    def update(self, query_key, memories, memory_rwds, prob):

        # if less than 2 items are returned, there is no leaf update, because the ranking is determined
        # what we need to do is compute the prediction error for each of the 2 items returned
        # if the prediction error for the first item is the same as the prediction error for the second item, no update (!!!)
        # if the prediction error for the first item is larger than the prediction error for the second item, update to prefer first > second
        #    update importance can be the _absolute difference in prediction error_
        # similarly, but the other way update to prefer second > first

        if len(memories) < 2: return

        memory_keys = [ mem[0] for mem in memories ]
        memory_keys,memory_rwds = zip(*sorted(zip(memory_keys,memory_rwds), key= lambda t: t[1], reverse=True))

        mem_0_advantage = memory_rwds[0] - memory_rwds[1]

        self.t += 1

        base    = 1-2*self.base.calculate_base(query_key, memory_keys[0])
        example = self.example.make_example(self.vw, query_key, memory_keys[0], base, 1, mem_0_advantage/prob)
        self.vw.learn(example)
        self.vw.finish_example(example)

        base    = 1-2*self.base.calculate_base(query_key, memory_keys[1]) 
        example = self.example.make_example(self.vw, query_key, memory_keys[1], base, -1, mem_0_advantage/prob)
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
