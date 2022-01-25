import time
import math

from itertools import count
from abc import ABC, abstractmethod
from typing import Sequence, Tuple

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

    def __init__(self, power_t:int, X: Sequence[str], initial_weight:float, base:str, learning_rate:float, l2:float, sgd:str, bl:str):
        
        self.args = (power_t,X,initial_weight,base,learning_rate,l2,sgd,bl)
        options = [
            "--quiet",
            f"-b {bits}",
            f"--l2 {l2}",
            f"--power_t {power_t}",
            f"--initial_weight {initial_weight}",
            "--noconstant",
            "--loss_function squared",
            "--min_prediction 0",
            "--max_prediction 40",
            f"--learning_rate {learning_rate:0.9f}",
            *[f"--interactions {x}" for x in X]
        ]

        if sgd == "coin":
            options += ["--coin" ]
        elif sgd == "not-norm":
            options += ["--sgd", "--adaptive", "--invariant"]
        elif sgd=="none":
            options += []
        else:
            raise Exception('unrecognized sgd parameter')

        self._base = base
        self._bl   = bl
        self.vw    = VowpalMediator().init_learner(" ".join(options), 0)
        self.t     = 0
        self.rng   = CobaRandom(1)

    def __reduce__(self):
        return (type(self), self.args)

    def predict(self, query_key, memory_keys):

        values = []
        for memory_key in memory_keys:
            example, base = self._make_example(query_key, memory_key, None, None)
            values.append(base+self.vw.predict(example))

        return values

    def update(self, query_key, memory_keys, memory_errs, weight):

        assert len(memory_keys) == len(memory_errs)
        if len(memory_keys) < 2: return

        tie_breakers = (self.rng.random() for _ in count())

        ranked_preferences = self.predict(query_key, memory_keys)
        best_preferrence   = list(sorted(zip(ranked_preferences, memory_errs, tie_breakers, memory_keys)))[0]
        
        ranked_alternatives = list(sorted(zip(memory_errs, ranked_preferences, tie_breakers, memory_keys)))
        best_alternative    = ranked_alternatives[1 if ranked_alternatives[0][3] is best_preferrence[3] else 0]

        # "what the scorer wanted to do": scorers preferred memory for query
        preferred_key = best_preferrence[3]
        preferred_err = best_preferrence[1]

        # "what the scorer could have done better": best memory for query when memory != preferred
        alternative_key = best_alternative[3]
        alternative_err = best_alternative[0]

        preferred_label   = 0 if preferred_err < alternative_err else 1
        alternative_label = 0 if alternative_err < preferred_err else 1
        update_weight     = weight*abs(preferred_err - alternative_err)

        if update_weight == 0: return

        self.t += 1

        examples = [
            self._make_example(query_key, preferred_key, preferred_label, update_weight)[0],
            self._make_example(query_key, alternative_key, alternative_label, update_weight)[0]
        ]

        for example in self.rng.shuffle(examples): self.vw.learn(example)

    def _diff_features(self, x1, x2):

        x1 = x1
        x2 = x2

        if isinstance(x1,dict) and isinstance(x2,dict):
            return { k:abs(x1.get(k,0)-x2.get(k,0)) for k in x1.keys() | x2.keys() }
        else:
            return [ abs(v1-v2) for v1,v2 in zip(x1,x2) ]

    def _make_example(self, query_key, memory_key, label, weight) -> Tuple[pyvw.example,float]:

        diff_x = self._diff_features(query_key.context, memory_key.context)
        diff_a = self._diff_features(query_key.action, memory_key.action)

        if self._base == "none":
            base = 0
        elif self._base == "l1":
            base = self._l1_norm(diff_x, diff_a)
        elif self._base == "l2":
            base = self._l2_norm(diff_x, diff_a)
        elif self._base == "cos":
            base = self._cos_dist(query_key, memory_key)
            assert 0 <= base and base <= 1
        elif self._base == "exp":
            base = 1-math.exp(-self._l2_norm(diff_x, diff_a))
            assert 0 <= base and base <= 1        
        else:
            raise Exception("Unrecognized Base")

        if query_key == memory_key:
            assert base == 0

        label = f"{0 if label is None else label} {0 if weight is None else weight} {base if self._bl == 'internal' else 0}"
        
        example = self.vw.make_example({'x': diff_x, 'a': diff_a}, label)

        #example.get_feature_number()
        #list(example.iter_features())
        #[ (i,self.vw.get_weight(i)) for i in reversed(range(self.vw.num_weights())) if self.vw.get_weight(i) != 0]

        return example, (base if self._bl == 'external' else 0)

    def _cos_dist(self, query_key, memory_key) -> float:

        n1 = self._dot(query_key, query_key)
        n2 = self._dot(memory_key, memory_key)

        return (1-self._dot(memory_key,query_key)/math.sqrt(n1*n2))/2

    def _l2_norm(self, c, a) -> float:
        return math.sqrt(self._dot(a,a) + self._dot(c,c))

    def _l1_norm(self, c, a) -> float:
        return sum([ sum(f.values()) if isinstance(f,dict) else sum(f) for f in [a,c] ])

    def _dot(self, x1,x2):
        if isinstance(x1, dict):
            return sum([x1[k]*x2[k] for k in (x1.keys() & x2.keys())])
        elif isinstance(x1, (tuple,list)):
            return sum([i*j for i,j in zip(x1,x2)])
        else:
            return self._dot(x1.context, x2.context) + self._dot(x1.action, x2.action)

    def __repr__(self) -> str:
        return f"rank{self.args}"

    def __str__(self) -> str:
        return self.__repr__()

class MetricScorer(Scorer):

    def __init__(self, metric) -> None:
        self._metric = metric

    def predict(self, query_key, memory_keys):

        if self._metric == "l2":
            return [self._l2_dist(query_key, memory_key) for memory_key in memory_keys]

        if self._metric == "l1":
            return [ self._l1_dist(query_key, memory_key) for memory_key in memory_keys ]

        if self._metric == "cos":
            return [ self._cos_dist(query_key, memory_key) for memory_key in memory_keys ]

        if self._metric == "exp":
            return [ 1-math.exp(-self._l2_dist(query_key, memory_key)) for memory_key in memory_keys ]

    def update(self, query_key, memory_keys, memory_errs, weight):
        pass

    def _diff_features(self, x1, x2):

        x1 = x1
        x2 = x2

        if isinstance(x1,dict) and isinstance(x2,dict):
            return { k:abs(x1.get(k,0)-x2.get(k,0)) for k in x1.keys() | x2.keys() }
        else:
            return [ abs(v1-v2) for v1,v2 in zip(x1,x2) ]

    def _dot(self, x1,x2):

        if isinstance(x1, dict):
            return sum([x1[k]*x2[k] for k in (x1.keys() & x2.keys())])
        elif isinstance(x1, (tuple,list)):
            return sum([i*j for i,j in zip(x1,x2)])
        else:
            return self._dot(x1.context, x2.context) + self._dot(x1.action, x2.action)

    def _l2_dist(self, query_key, memory_key) -> float:

        a = self._diff_features(query_key.action, memory_key.action)
        c = self._diff_features(query_key.context, memory_key.context)

        return math.sqrt(self._dot(a,a) + self._dot(c,c))

    def _l1_dist(self, query_key, memory_key) -> float:

        a = self._diff_features(query_key.action, memory_key.action)
        c = self._diff_features(query_key.context, memory_key.context)

        return sum([ sum(f.values()) if isinstance(f,dict) else sum(f) for f in [a,c] ])

    def _cos_dist(self, query_key, memory_key) -> float:

        n1 = self._dot(query_key, query_key)
        n2 = self._dot(memory_key, memory_key)

        return (1-self._dot(memory_key,query_key)/math.sqrt(n1*n2))/2

    def __repr__(self) -> str:
        return f"metric({self._metric})"

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
