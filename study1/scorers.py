import time
import math
import operator
import itertools

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

    def __init__(self, base:str, X: Sequence[str] = ['x','a'], F:Sequence[str] = None):

        self.args = (base,X,F)

        options = [
            "--quiet",
            f"-b {bits}",
            f"--power_t {0}",
            f"--random_seed {1}",
            "--coin",
            "--noconstant",
            "--loss_function squared",
            "--min_prediction 0",
            "--max_prediction 3"
        ]

        self.times = [0,0,0]

        self._X    = tuple(X)
        self._F    = tuple(F or X)
        self._base = base
        self.vw    = VowpalMediator().init_learner(" ".join(options), 1)
        self.t     = 0
        self.rng   = CobaRandom(1)
        self._weights = []

    def __reduce__(self):
        return (type(self), self.args)

    def predict(self, query_key, memory_keys):
        return [ 1-math.exp(-self._predict(query_key,key)) for key in memory_keys]

    def update(self, query_key, memory_keys, memory_errs, weight):

        assert len(memory_keys) == len(memory_errs)
        if len(memory_keys) <2 : return

        tie_breakers = self.rng.randoms(len(memory_keys))
        scores       = self.predict(query_key, memory_keys)

        top1_by_score = list(sorted(zip(scores, memory_errs, tie_breakers, memory_keys)))[0]
        top2_by_error = list(sorted(zip(memory_errs, scores, tie_breakers, memory_keys)))[0:2]

        best_alternative = top2_by_error[0 if top2_by_error[1][3] is top1_by_score[3] else 1]

        # "what the scorer wanted to do": scorers preferred memory for query
        preferred_key = top1_by_score[3]
        preferred_err = top1_by_score[1]

        # "what the scorer could have done better": best memory for query when memory != preferred
        alternative_key = best_alternative[3]
        alternative_err = best_alternative[0]

        preferred_label   = 0 if preferred_err < alternative_err else 1
        alternative_label = 0 if alternative_err < preferred_err else 1
        update_weight     = weight*abs(preferred_err - alternative_err)

        if update_weight == 0: return

        self.t += 1

        examples = [
            [query_key, preferred_key, preferred_label, update_weight],
            [query_key, alternative_key, alternative_label, update_weight]
        ]

        for example in self.rng.shuffle(examples): 
            self._learn(*example)

        self._weights = []

    def _bns(self, key1, key2) -> Tuple[float, dict]:
        diff_x = self._sub(key1.raw(self._X), key2.raw(self._X))
        diff_f = diff_x if self._X == self._F else self._sub(key1.raw(self._F), key2.raw(self._F))

        if self._base == "none":
            base = 0
        elif self._base == "l1":
            base = self._l1_norm(diff_f)
        elif self._base == "l2":
            base = self._l2_norm(diff_f)
        elif self._base == "cos":
            base = self._cos_dist(key1, key2)
            assert 0 <= base and base <= 1
        elif self._base == "exp":
            #l1_norm is about 8x faster than l2_norm in isolation...
            #end to end l1_norm gives a decrease in run time of approx 25% compared to l2_norm
            base = 1-math.exp(-self._l1_norm(diff_f))
            assert 0 <= base and base <= 1
        else:
            raise Exception("Unrecognized Base")

        if key1 == key2: assert base == 0

        return base, {'x': diff_x }

    def _predict(self, key1, key2):
        start = time.time()
        base, ns = self._bns(key1,key2)
        self.times[0] += time.time()-start

        start = time.time()
        pred = self.vw.predict(self.vw.make_example(ns, f"{0} {0} {base}"))
        self.times[1] += time.time()-start

        return pred

    def _learn(self, key1, key2, label, weight) -> pyvw.example:
        base, ns = self._bns(key1,key2)
        self.vw.learn(self.vw.make_example(ns, f"{label} {weight} {base}"))

    def _cos_dist(self, key1, key2) -> float:
        x1 = key1.raw(self._F)
        x2 = key2.raw(self._F)
        return (1-self._dot2(x1,x2)/math.sqrt(self._dot1(x1)*self._dot1(x2)))/2

    def _l2_norm(self, x) -> float:
        return math.sqrt(self._dot1(x))

    def _l1_norm(self, x) -> float:
        return sum([abs(v) for v in x.values()]) if isinstance(x,dict) else sum([abs(v) for v in x])

    def _dot1(self, x):
        values = x.values() if isinstance(x,dict) else x
        return sum(map(operator.mul,values,values))

    def _dot2(self, x1, x2):
        if isinstance(x1, dict):
            return sum(x1[k]*x2[k] for k in (x1.keys() & x2.keys()))
        else:
            return sum(i*j for i,j in zip(x1,x2))

    def _sub(self, x1, x2):
        if isinstance(x1,dict):
            short = x1 if len(x1) < len(x2) else x2
            long  = x2 if len(x1) < len(x2) else x1

            sub = dict(long)

            for key,val in short.items():
                if key in sub:
                    sub[key] = abs(sub[key]-val)
                else:
                    sub[key] = val
            return sub

            #return { k:abs(x1.get(k,0)-x2.get(k,0)) for k in x1.keys() | x2.keys() }
        else:
            return [ abs(v1-v2) for v1,v2 in zip(x1,x2) ]

    def __repr__(self) -> str:
        return f"Rank{self.args}"

    def __str__(self) -> str:
        return self.__repr__()

class RankScorer2(Scorer):

    def __init__(self, base:str, X: Sequence[str] = ['x','a'], F:Sequence[str] = None):

        self.args = (base,X,F)

        options = [
            "--quiet",
            f"-b {bits}",
            f"--power_t {0}",
            f"--random_seed {1}",
            "--coin",
            "--noconstant",
            "--loss_function squared",
            "--min_prediction 0",
            "--max_prediction 100",
            "--interactions xx",
            "--interactions zz",
            "--interactions xz",
        ]

        self._X    = tuple(X)
        self._F    = tuple(F or X)
        self._base = base
        self.vw    = VowpalMediator().init_learner(" ".join(options), 1)
        self.t     = 0
        self.rng   = CobaRandom(1)
        self._weights = []

    def __reduce__(self):
        return (type(self), self.args)

    def predict(self, query_key, memory_keys):
        return [ 1-math.exp(-self._predict(query_key,key)) for key in memory_keys]

    def update(self, query_key, memory_keys, memory_errs, weight):

        assert len(memory_keys) == len(memory_errs)
        if len(memory_keys) <2 : return

        tie_breakers = self.rng.randoms(len(memory_keys))
        scores       = self.predict(query_key, memory_keys)

        top1_by_score = list(sorted(zip(scores, memory_errs, tie_breakers, memory_keys)))[0]
        top2_by_error = list(sorted(zip(memory_errs, scores, tie_breakers, memory_keys)))[0:2]

        best_alternative = top2_by_error[0 if top2_by_error[1][3] is top1_by_score[3] else 1]

        # "what the scorer wanted to do": scorers preferred memory for query
        preferred_key = top1_by_score[3]
        preferred_err = top1_by_score[1]

        # "what the scorer could have done better": best memory for query when memory != preferred
        alternative_key = best_alternative[3]
        alternative_err = best_alternative[0]

        preferred_label   = 0 if preferred_err < alternative_err else 1
        alternative_label = 0 if alternative_err < preferred_err else 1
        update_weight     = weight*abs(preferred_err - alternative_err)

        if update_weight == 0: return

        self.t += 1

        examples = [
            [query_key, preferred_key, preferred_label, update_weight],
            [query_key, alternative_key, alternative_label, update_weight]
        ]

        for example in self.rng.shuffle(examples): 
            self._learn(*example)

        self._weights = []

    def _bns(self, key1, key2) -> Tuple[float, dict]:
        diff_x = self._sub(key1.raw(self._X), key2.raw(self._X))
        diff_f = diff_x if self._X == self._F else self._sub(key1.raw(self._F), key2.raw(self._F))

        if self._base == "none":
            base = 0
        elif self._base == "l1":
            base = self._l1_norm(diff_f)
        elif self._base == "l2":
            base = self._l2_norm(diff_f)
        elif self._base == "cos":
            base = self._cos_dist(key1, key2)
            assert 0 <= base and base <= 1
        elif self._base == "exp":
            base = 1-math.exp(-self._l1_norm(diff_f))
            assert 0 <= base and base <= 1
        else:
            raise Exception("Unrecognized Base")

        if key1 == key2: assert base == 0

        return base, {'x': key1.raw(self._X), 'z': key2.raw(self._X) }

    def _predict(self, key1, key2):
        base, ns = self._bns(key1,key2)
        pred = self.vw.predict(self.vw.make_example(ns, f"{0} {0} {base}"))
        return pred

    def _learn(self, key1, key2, label, weight) -> pyvw.example:
        base, ns = self._bns(key1,key2)
        self.vw.learn(self.vw.make_example(ns, f"{label} {weight} {base}"))

    def _cos_dist(self, query_key, memory_key) -> float:
        x1 = query_key.raw(self._F)
        x2 = memory_key.raw(self._F)
        return (1-self._dot(x1,x2)/math.sqrt(self._dot(x1, x1)*self._dot(x2, x2)))/2

    def _l2_norm(self, x) -> float:
        return math.sqrt(self._dot(x,x))

    def _l1_norm(self, x) -> float:
        return sum([abs(v) for v in x.values()]) if isinstance(x,dict) else sum([abs(v) for v in x])

    def _dot(self, x1, x2):
        if isinstance(x1, dict):
            return sum(x1[k]*x2[k] for k in (x1.keys() & x2.keys()))
        else:
            return sum(i*j for i,j in zip(x1,x2))

    def _sub(self, x1, x2):
        if isinstance(x1,dict):
            return { k:abs(x1.get(k,0)-x2.get(k,0)) for k in x1.keys() | x2.keys() }
        else:
            return [ abs(v1-v2) for v1,v2 in zip(x1,x2) ]

    def __repr__(self) -> str:
        return f"Rank2{self.args}"

    def __str__(self) -> str:
        return self.__repr__()



class DistScorer(Scorer):

    def __init__(self, metric, features=['x','a']) -> None:
        self._metric = metric
        self._features = tuple(features)

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

    def _sub(self, x1, x2):
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
            return self._dot(x1.raw(self._features), x2.raw(self._features))

    def _l2_dist(self, query_key, memory_key) -> float:
        x = self._sub(query_key.raw(self._features), memory_key.raw(self._features))
        return math.sqrt(self._dot(x,x))

    def _l1_dist(self, query_key, memory_key) -> float:
        x = self._sub(query_key.raw(self._features), memory_key.raw(self._features))
        return sum(x.values()) if isinstance(x,dict) else sum(x)

    def _cos_dist(self, query_key, memory_key) -> float:
        n1 = self._dot(query_key, query_key)
        n2 = self._dot(memory_key, memory_key)

        return (1-self._dot(memory_key,query_key)/math.sqrt(n1*n2))/2

    def __repr__(self) -> str:
        return f"dist({self._metric})"

    def __str__(self) -> str:
        return self.__repr__()

class RandomScorer(Scorer):

    def __init__(self):
        self.rng = CobaRandom(1)

    def predict(self, query_key, memory_keys):
        return self.rng.randoms(len(memory_keys))

    def update(self, *args):
        pass

    def __repr__(self) -> str:
        return f"rand"

    def __str__(self) -> str:
        return self.__repr__()
