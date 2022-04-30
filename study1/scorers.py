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

    def __init__(self, base:str, X: Sequence[str] = []):

        self.args = (base,X)

        X = X or ['x','a']

        options = [
            "--quiet",
            f"-b {bits}",
            f"--power_t {0}",
            f"--random_seed {1}",
            "--coin",
            "--noconstant",
            "--loss_function squared",
            "--min_prediction 0",
            "--max_prediction 2",
        ]

        self._X    = tuple(X)
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
            self._make_example(query_key, preferred_key, preferred_label, update_weight),
            self._make_example(query_key, alternative_key, alternative_label, update_weight)
        ]

        for example in self.rng.shuffle(examples): self.vw.learn(example)
        self._weights = []

    def _sub(self, x1, x2):
        if isinstance(x1,dict) and isinstance(x2,dict):
            return { k:abs(x1.get(k,0)-x2.get(k,0)) for k in x1.keys() | x2.keys() }
        else:
            return [ abs(v1-v2) for v1,v2 in zip(x1,x2) ]

    def _predict(self, key1,key2):

        diff_x = key1.raw(self._X), key2.raw(self._X)
        diff_x = self._sub(key1.raw(self._X), key2.raw(self._X))

        if self._base == "none":
            base = 0
        elif self._base == "l1":
            base = self._l1_norm(diff_x)
        elif self._base == "l2":
            base = self._l2_norm(diff_x)
        elif self._base == "cos":
            base = self._cos_dist(key1, key2)
            assert 0 <= base and base <= 1
        elif self._base == "exp":
            base = 1-math.exp(-self._l2_norm(diff_x))
            assert 0 <= base and base <= 1
        else:
            raise Exception("Unrecognized Base")

        if key1 == key2:
            assert base == 0

        if isinstance(diff_x,list):
            self._weights = self._weights or [self.vw._vw.get_weight(i) for i in range(len(diff_x))]
            return base + sum(w*f for w,f in zip(self._weights,diff_x) if f != 0)
        else:
            return base + self.vw.predict(self.vw.make_example({'x': diff_x }, None))

    def _make_example(self, query_key, memory_key, label, weight) -> pyvw.example:

        diff_x = self._sub(query_key.raw(self._X), memory_key.raw(self._X))

        if self._base == "none":
            base = 0
        elif self._base == "l1":
            base = self._l1_norm(diff_x)
        elif self._base == "l2":
            base = self._l2_norm(diff_x)
        elif self._base == "cos":
            base = self._cos_dist(query_key, memory_key)
            assert 0 <= base and base <= 1
        elif self._base == "exp":
            base = 1-math.exp(-self._l2_norm(diff_x))
            assert 0 <= base and base <= 1
        else:
            raise Exception("Unrecognized Base")

        if query_key == memory_key:
            assert base == 0

        label = f"{0 if label is None else label} {0 if weight is None else weight} {base}"
        
        return self.vw.make_example({'x': diff_x }, label)

        #example.get_feature_number()
        #list(example.iter_features())
        #[ (i,self.vw.get_weight(i)) for i in reversed(range(self.vw.num_weights())) if self.vw.get_weight(i) != 0]

    def _cos_dist(self, query_key, memory_key) -> float:
        x1 = query_key.raw(self._X)
        x2 = memory_key.raw(self._X)
        return (1-self._dot(x1,x2)/math.sqrt(self._dot(x1, x1)*self._dot(x2, x2)))/2

    def _l2_norm(self, x) -> float:
        return math.sqrt(self._dot(x,x))

    def _l1_norm(self, x) -> float:
        return sum(x.values()) if isinstance(x,dict) else sum(x)

    def _dot(self, x1,x2):
        if isinstance(x1, dict):
            return sum(x1[k]*x2[k] for k in (x1.keys() & x2.keys()))
        elif isinstance(x1, (tuple,list)):
            return sum(i*j for i,j in zip(x1,x2))

    def __repr__(self) -> str:
        return f"rank{self.args}"

    def __str__(self) -> str:
        return self.__repr__()

class RankScorer2(Scorer):

    def __init__(self, base:str, X: Sequence[str] = [],v2=False):

        self.args = (base,X,v2)

        X = X or ['x','a']

        options = [
            "--quiet",
            f"-b {bits}",
            f"--power_t {0}",
            f"--random_seed {1}",
            "--coin",
            "--noconstant",
            "--loss_function squared",
            "--min_prediction -2",
            "--max_prediction 2",
        ]

        if not v2:
            if 'x' not in X: options.append("--ignore_linear x")
            if 'a' not in X: options.append("--ignore_linear a")
            options.extend([f"--interactions {x}" for x in X if len(x) > 1])

        self._X    = X
        self._base = base
        self._v2   = v2
        self.vw    = VowpalMediator().init_learner(" ".join(options), 1)
        self.t     = 0
        self.rng   = CobaRandom(1)

    def __reduce__(self):
        return (type(self), self.args)

    def predict(self, query_key, memory_keys):
        return [ 1-math.exp(-self.vw.predict(self._make_example(query_key, key, None, None))) for key in memory_keys]

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
            self._make_example(query_key, preferred_key, preferred_label, update_weight),
            self._make_example(query_key, alternative_key, alternative_label, update_weight)
        ]
        
        for example in self.rng.shuffle(examples): self.vw.learn(example)

    def _sub(self, x1, x2):

        x1 = x1
        x2 = x2

        if isinstance(x1,dict) and isinstance(x2,dict):
            return { k:abs(x1.get(k,0)-x2.get(k,0)) for k in x1.keys() | x2.keys() }
        else:
            return [ v1-v2 for v1,v2 in zip(x1,x2) ]

    def _make_example(self, query_key, memory_key, label, weight) -> pyvw.example:

        if not self._v2:
            diff_x = self._sub(query_key.raw(['x']), memory_key.raw(['x']))
            diff_a = self._sub(query_key.raw(['a']), memory_key.raw(['a']))
        else:
            diff_x = self._sub(query_key.raw(self._X), memory_key.raw(self._X))
            diff_a = []

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

        label = f"{0 if label is None else label-base} {0 if weight is None else weight} {base}"
        
        if not self._v2:
            example = self.vw.make_example({'x': diff_x, 'a': diff_a}, label)
        else:
            example = self.vw.make_example({'x': diff_x }, label)

        #example.get_feature_number()
        #list(example.iter_features())
        #[ (i,self.vw.get_weight(i)) for i in reversed(range(self.vw.num_weights())) if self.vw.get_weight(i) != 0]

        return example

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
            return self._dot(x1.raw(['x']), x2.raw(['x'])) + self._dot(x1.raw(['a']), x2.raw(['a']))

    def __repr__(self) -> str:
        return f"rank2{self.args}"

    def __str__(self) -> str:
        return self.__repr__()

class RankScorer3(Scorer):

    def __init__(self, base:str, X: Sequence[str] = [],v2=False):

        self.args = (base,X,v2)

        X = X or ['x','a']

        options = [
            "--quiet",
            f"-b {bits}",
            f"--power_t {0}",
            f"--random_seed {1}",
            "--coin",
            "--noconstant",
            "--loss_function squared",
            "--min_prediction -2",
            "--max_prediction 2",
        ]

        if not v2:
            if 'x' not in X: options.append("--ignore_linear x")
            if 'a' not in X: options.append("--ignore_linear a")
            options.extend([f"--interactions {x}" for x in X if len(x) > 1])

        self._X    = X
        self._base = base
        self._v2   = v2
        self.vw    = VowpalMediator().init_learner(" ".join(options), 1)
        self.t     = 0
        self.rng   = CobaRandom(1)

    def __reduce__(self):
        return (type(self), self.args)

    def predict(self, query_key, memory_keys):
        return [ 1-math.exp(-self.vw.predict(self._make_example(query_key, key, None, None))) for key in memory_keys]

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
            self._make_example(query_key, preferred_key, preferred_label, update_weight),
            self._make_example(query_key, alternative_key, alternative_label, update_weight)
        ]
        
        for example in self.rng.shuffle(examples): self.vw.learn(example)

    def _sub(self, x1, x2):

        x1 = x1
        x2 = x2

        if isinstance(x1,dict) and isinstance(x2,dict):
            return { k:abs(x1.get(k,0)-x2.get(k,0)) for k in x1.keys() | x2.keys() }
        else:
            return [ abs(v1-v2) for v1,v2 in zip(x1,x2) ]

    def _make_example(self, query_key, memory_key, label, weight) -> pyvw.example:

        if not self._v2:
            diff_x = self._sub(query_key.raw(['x']), memory_key.raw(['x']))
            diff_a = self._sub(query_key.raw(['a']), memory_key.raw(['a']))
        else:
            diff_x = self._sub(query_key.raw(self._X), memory_key.raw(self._X))
            diff_a = []

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

        label = f"{0 if label is None else label} {0 if weight is None else weight}"

        if not self._v2:
            example = self.vw.make_example({'x': diff_x, 'a': diff_a}, label)
        else:
            example = self.vw.make_example({'x': diff_x, 'z':base}, label)

        #example.get_feature_number()
        #list(example.iter_features())
        #[ (i,self.vw.get_weight(i)) for i in reversed(range(self.vw.num_weights())) if self.vw.get_weight(i) != 0]

        return example

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
            return self._dot(x1.raw(['x']), x2.raw(['x'])) + self._dot(x1.raw(['a']), x2.raw(['a']))

    def __repr__(self) -> str:
        return f"rank3{self.args}"

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
        return f"metric({self._metric})"

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
