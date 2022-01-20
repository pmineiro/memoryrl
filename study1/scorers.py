import time
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

    times = [0,0]

    def __init__(self, power_t:int, X: Sequence[str], initial_weight:float, coin:bool, base:str):
        
        self.args = (power_t,X,initial_weight,coin,base)
        options = [
            "--quiet",
            f"-b {bits}",
            f"--power_t {power_t}",
            f"--initial_weight {initial_weight}",
            "--coin" if coin else "",
            "--noconstant",
            "--loss_function squared",
            "--min_prediction 0",
            "--max_prediction 1",
            *[f"--interactions {x}" for x in X]
        ]
        
        self.vw      = VowpalMediator().init_learner(" ".join(options), 0)
        self.t       = 0
        self.power_t = power_t
        self.rng     = CobaRandom(1)

    def __reduce__(self):
        return (type(self), self.args)

    def predict(self, query_key, memory_keys):

        values = []
        for memory_key in memory_keys:
            values.append(self.vw.predict(self._make_example(query_key, memory_key, None, None)))

            #assert values[-1] - max(0,sum([self.vw.get_weight(i)*v for i,v in example.iter_features()])) < .00001

        #if query_key in memory_keys and len([v for v in values if v == 0]) == 1:

        # if query_key in memory_keys:
        #     assert values[memory_keys.index(query_key)] == 0

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

        if preferred_advantage == 0 or weight == 0: return

        self.t += 1

        preferred_label   = 0 if preferred_advantage < 0 else 1
        alternative_label = 1-preferred_label
        update_weight     = weight*abs(preferred_advantage)

        if self.rng.random() < .5:
            self.vw.learn(self._make_example(query_key, preferred_key, preferred_label, update_weight))
            self.vw.learn(self._make_example(query_key, alternative_key, alternative_label, update_weight))
        else:
            self.vw.learn(self._make_example(query_key, alternative_key, alternative_label, update_weight))
            self.vw.learn(self._make_example(query_key, preferred_key, preferred_label, update_weight))

    def _diff_features(self, x1, x2):

        x1 = x1
        x2 = x2

        if isinstance(x1,dict) and isinstance(x2,dict):
            return { k:abs(x1.get(k,0)-x2.get(k,0)) for k in x1.keys() | x2.keys() }
        else:
            return [ abs(v1-v2) for v1,v2 in zip(x1,x2) ]

    def _make_example(self, query_key, memory_key, label, weight) -> pyvw.example:

        start_time = time.time()
        diff_a = self._diff_features(query_key.action, memory_key.action)
        diff_x = self._diff_features(query_key.context, memory_key.context)
        RankScorer.times[0] += time.time()-start_time

        if self.args[4]=="none":
            base = 0
        elif self.args[4] == "l2":
            base = self._l2_dist(query_key, memory_key)
        elif self.args[4] == "cos":
            base = self._cos_dist(query_key, memory_key)
            assert 0 <= base and base <= 1
        elif self.args[4] == "exp":
            base = 1-math.exp(-self._l2_dist(query_key, memory_key))
            assert 0 <= base and base <= 1        
        else:
            raise Exception("Unrecognized Base")

        if query_key == memory_key:
            assert base == 0

        start_time = time.time()
        label = f"{0 if label is None else label} {1 if weight is None else weight} {base}"
        example = self.vw.make_example({'x': diff_x, 'a': diff_a}, label)
        RankScorer.times[1] += time.time()-start_time

        #example.get_feature_number()
        #list(example.iter_features())
        #[ (i,self.vw.get_weight(i)) for i in reversed(range(self.vw.num_weights())) if self.vw.get_weight(i) != 0]

        return example

    def _cos_dist(self, query_key, memory_key) -> float:

        n1 = self._dot(query_key, query_key)
        n2 = self._dot(memory_key, memory_key)

        return (1-self._dot(memory_key,query_key)/math.sqrt(n1*n2))/2

    def _l2_dist(self, query_key, memory_key) -> float:

        a = self._diff_features(query_key.action, memory_key.action)
        c = self._diff_features(query_key.context, memory_key.context)

        return math.sqrt(self._dot(a,a) + self._dot(c,c))

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
