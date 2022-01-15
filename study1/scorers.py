from abc import ABC, abstractmethod
from typing import Sequence

from coba.random import CobaRandom
from coba.learners.vowpal import VowpalMediator
from coba.pipes.filters import Flatten

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

    def __init__(self, power_t:int=0, X: Sequence[str]=[]):
        
        interactions = " ".join(f"--interactions {x}" for x in X)
        self.vw = pyvw.vw(f'--quiet -b {bits} --power_t {power_t} --noconstant --loss_function squared --min_prediction 0 --max_prediction 1 {interactions}')

        self.t       = 0
        self.power_t = power_t
        self.rng     = CobaRandom(1)

        self.args = (power_t,X)

    def __reduce__(self):
        return (type(self), self.args)

    def predict(self, query_key, memory_keys):

        values = []

        for memory_key in memory_keys:
            example = self._make_example(query_key, memory_key)
            values.append(self.vw.predict(example))
            self.vw.finish_example(example)

        # if query_key in memory_keys and len([v for v in values if v == 0]) == 1:
        #     assert memory_keys.index(query_key) == values.index(0)

        # if query_key in memory_keys and len([v for v in values if v == 0]) > 1:
        #     print(f"interesting {len([v for v in values if v == 0])} {self.t} {len([ self.vw.get_weight(i) for i in reversed(range(self.vw.num_weights())) if self.vw.get_weight(i) < 0])}")

        return values

    def update(self, query_key, memory_keys, memory_errs):

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

        example = self._make_example(query_key, preferred_key, preferred_label, abs(preferred_advantage))
        self.vw.learn(example)
        self.vw.finish_example(example)

        example = self._make_example(query_key, alternative_key, alternative_label, abs(preferred_advantage))
        self.vw.learn(example)
        self.vw.finish_example(example)

    def _diff_features(self, x1, x2):
        
        x1 = x1
        x2 = x2

        if isinstance(x1,dict) and isinstance(x2,dict):
            return { k:abs(x1.get(k,0)-x2.get(k,0)) for k in x1.keys() | x2.keys() }
        else:
            return [ abs(v1-v2) for v1,v2 in zip(x1,x2) ]

    def _make_example(self, query_key, memory_key, label = None, weight = 1) -> pyvw.example:

        a = VowpalMediator.prep_features(self._diff_features(query_key.action, memory_key.action))
        x = VowpalMediator.prep_features(self._diff_features(query_key.context, memory_key.context))

        example = pyvw.example(self.vw, {'x': x, 'a': a})

        if label is not None:
            example.set_label_string(f"{label} {weight}")
        
        #example.get_feature_number()
        #list(example.iter_features())
        #[ (i,self.vw.get_weight(i)) for i in reversed(range(self.vw.num_weights())) if self.vw.get_weight(i) != 0]

        return example

    def __repr__(self) -> str:
        if not self.args[1]:
            return f"rank({self.args[0]})"
        else:
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
