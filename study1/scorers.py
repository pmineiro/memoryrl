import math

from abc import ABC, abstractmethod
from typing import Sequence

from coba.random import CobaRandom
from coba.learners.vowpal import VowpalMediator

from vowpalwabbit import pyvw

bits = 20

class Scorer(ABC):

    @abstractmethod
    def predict(self, query_key, memory_keys) -> Sequence[float]:
        ...

    @abstractmethod
    def update(self, query_key, memory_keys, memory_errs):
        ...

class RankScorer(Scorer):

    def __init__(self, power_t:int, X: Sequence[str], initial_weight:float, coin:bool):
        
        self.args = (power_t,X,initial_weight,coin)

        interactions = " ".join(f"--interactions {x}" for x in X)
        coin_flag    = "--coin" if coin else ""
        self.vw = pyvw.vw(f'--quiet -b {bits} --power_t {power_t} --initial_weight {initial_weight} {coin_flag} --noconstant --loss_function squared --min_prediction 0 --max_prediction 1 {interactions}')

        self.t       = 0
        self.power_t = power_t
        self.rng     = CobaRandom(1)

    def __reduce__(self):
        return (type(self), self.args)

    def predict(self, query_key, memory_keys):

        values = []

        for memory_key in memory_keys:
            example = self._make_example(query_key, memory_key, None, None)
            values.append(self.vw.predict(example))
            self.vw.finish_example(example)

        # if query_key in memory_keys and len([v for v in values if v == 0]) == 1:
        #     assert memory_keys.index(query_key) == values.index(0)

        # if query_key in memory_keys and len([v for v in values if v == 0]) > 1:
        #     print(f"interesting {len([v for v in values if v == 0])} {self.t} {len([ self.vw.get_weight(i) for i in reversed(range(self.vw.num_weights())) if self.vw.get_weight(i) < 0])}")

        return values

    def update(self, query_key, memory_keys, memory_errs, weight):

        assert len(memory_keys) == len(memory_errs)
        if len(memory_keys) < 2: return

        memory_keys, memory_errs = zip(*sorted(zip(memory_keys,memory_errs), key= lambda t: t[1]))
        scorer_preferences = self.predict(query_key, memory_keys)

        # "what the scorer wanted to do": prefereed_key
        preferred_key = min(zip(memory_keys,scorer_preferences), key=lambda t: t[1])[0]
        preferred_err = min(zip(memory_errs,scorer_preferences), key=lambda t: t[1])[0]

        # "what the scorer could have done better": best memory_rwds among memories != preferred_key
        alternative_key = memory_keys[0] if memory_keys[0] is not preferred_key else memory_keys[1] 
        alternative_err = memory_errs[0] if memory_keys[0] is not preferred_key else memory_errs[1] 

        preferred_advantage = preferred_err - alternative_err

        if preferred_advantage == 0: return

        self.t += 1

        preferred_label   = 0 if preferred_advantage < 0 else 1
        alternative_label = 1-preferred_label 

        example = self._make_example(query_key, preferred_key, preferred_label, weight*abs(preferred_advantage))
        self.vw.learn(example)
        self.vw.finish_example(example)

        example = self._make_example(query_key, alternative_key, alternative_label, weight*abs(preferred_advantage))
        self.vw.learn(example)
        self.vw.finish_example(example)

    def _diff_features(self, x1, x2):
        
        x1 = x1
        x2 = x2

        if isinstance(x1,dict) and isinstance(x2,dict):
            return { k:abs(x1.get(k,0)-x2.get(k,0)) for k in x1.keys() | x2.keys() }
        else:
            return [ abs(v1-v2) for v1,v2 in zip(x1,x2) ]

    def _make_example(self, query_key, memory_key, label, weight) -> pyvw.example:

        a = VowpalMediator.prep_features(self._diff_features(query_key.action, memory_key.action))
        x = VowpalMediator.prep_features(self._diff_features(query_key.context, memory_key.context))

        example = pyvw.example(self.vw, {'x': x, 'a': a})
        base = self._cos_dist(query_key, memory_key)
        
        assert 0 <= base and base <= 1

        example.set_label_string(f"{label or ''} {1 if weight is None else weight} {base}")
        
        #example.get_feature_number()
        #list(example.iter_features())
        #[ (i,self.vw.get_weight(i)) for i in reversed(range(self.vw.num_weights())) if self.vw.get_weight(i) != 0]

        return example

    def _cos_dist(self, query_key, memory_key) -> float:
        
        def _dot(x1,x2):

            if isinstance(x1, dict):
                return sum([x1[k]*x2[k] for k in (x1.keys() & x2.keys())])
            elif isinstance(x1, (tuple,list)):
                return sum([i*j for i,j in zip(x1,x2)])
            else:
                return _dot(x1.context, x2.context) + _dot(x1.action, x2.action)

        n1 = _dot(query_key,query_key)
        n2 = _dot(memory_key, memory_key)
        
        return (1-_dot(memory_key,query_key)/math.sqrt(n1*n2))/2

    def __repr__(self) -> str:
        return f"rank{self.args}"

    def __str__(self) -> str:
        return self.__repr__()

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