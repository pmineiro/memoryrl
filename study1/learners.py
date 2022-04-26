import time
import math

from typing import Hashable, Sequence, Dict, Any

from memory import CMT

from coba.learners import VowpalMediator
from coba.encodings import InteractionsEncoder

logn = 500
bits = 20

class MemoryKey:

    def __init__(self, context, action) -> None:

        self.x = context
        self.a = action
        
        self.raw_cache = {}
        self.np_cache = {}

        self._hash = hash((context,action))

    def raw(self, features):
        features = tuple(features)
        if features not in self.raw_cache:
            self.raw_cache[features] = InteractionsEncoder(features).encode(x=self.x,a=self.a)
        return self.raw_cache[features]

    def features(self, features):
        features = tuple(features)
        if features not in self.np_cache:
            raw_features=self.raw(features)
            if isinstance(raw_features,dict):
                from sklearn.feature_extraction import FeatureHasher
                self.np_cache[features] = FeatureHasher().fit_transform([raw_features])
            else:
                import numpy as np
                self.np_cache[features] = np.array([raw_features])
        return self.np_cache[features]

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, MemoryKey) and self.x == __o.x and self.a == __o.a

class MemorizedLearner1:

    def __init__(self, epsilon: float, cmt: CMT) -> None:

        assert 0 <= epsilon and epsilon <= 1

        self._epsilon = epsilon
        self._i       = 0
        self._cmt     = cmt
        self._times   = [0, 0]

    @property
    def params(self) -> Dict[str,Any]:
        return { 'family': 'memorized_taken1','e':self._epsilon, **self._cmt.params }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        self._i += 1

        if logn and self._i % logn == 0:
           print(f"MEM {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"MEM {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        predict_start = time.time()

        rewards = [ self._cmt.query(MemoryKey(context, a))[0] for a in actions]

        greedy_r = -math.inf
        greedy_A = []

        for action, mem_value in zip(actions, rewards):

            mem_value = mem_value or 0

            if mem_value == greedy_r:
                greedy_A.append(action)

            if mem_value > greedy_r:
                greedy_r = mem_value
                greedy_A = [action]

        self._times[0] += time.time()-predict_start

        min_p = self._epsilon / len(actions)
        grd_p = (1-self._epsilon)/len(greedy_A)

        return [ grd_p+min_p if a in greedy_A else min_p for a in actions ], len(actions)

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, predict_info: Any) -> None:
        """Learn about the result of an action that was taken in a context."""
        n_actions = predict_info

        memory_key = MemoryKey(context, action)

        self._cmt.update(key=memory_key, outcome=reward, weight=1/(n_actions*probability))

        learn_start = time.time()
        self._cmt.insert(key=memory_key, value=reward, weight=1/(n_actions*probability))
        self._times[1] += time.time()-learn_start

class MemorizedLearner2:

    def __init__(self, epsilon: float, cmt: CMT, X:str, coin:bool, constant:bool) -> None:

        assert 0 <= epsilon and epsilon <= 1

        self._epsilon = epsilon
        self._i       = 0
        self._cmt     = cmt
        self._times   = [0, 0]
        self._args    = (X, coin, constant)

        if X == 'xa':
            args = f"--quiet --cb_explore_adf --epsilon {epsilon} --ignore_linear x --interactions xa --random_seed {1}"

        if X == 'xxa':
            args = f"--quiet --cb_explore_adf --epsilon {epsilon} --ignore_linear x --interactions xa --interactions xxa --random_seed {1}"

        if coin: 
            args += ' --coin'

        if not constant:
            args += " --noconstant"

        self._vw = VowpalMediator().init_learner(args,4)

    @property
    def params(self) -> Dict[str,Any]:
        return { 'family': 'memorized_taken2', 'e': self._epsilon, **self._cmt.params, "other": self._args }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        self._i += 1

        if logn and self._i % logn == 0:
           print(f"MEM {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"MEM {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        predict_start = time.time()

        ms = [ self._cmt.query(MemoryKey(context, a)) for a in actions ]

        adfs = [ {'a':a, 'm':[m[0],m[1],m[0]*m[1]] }  for a,m in zip(actions,ms) ]
        probs = self._vw.predict(self._vw.make_examples({'x':self._flat(context)}, adfs, None))

        self._times[0] += time.time()-predict_start

        return probs, (actions,adfs)

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, predict_info: Any) -> None:
        """Learn about the result of an action that was taken in a context."""

        learn_start = time.time()

        actions,adfs = predict_info
        n_actions    = len(actions)

        memory_key = MemoryKey(context, action)

        self._cmt.update(key=memory_key, outcome=reward, weight=1/(n_actions*probability))
        self._cmt.insert(key=memory_key, value=reward, weight=1/(n_actions*probability))

        self._times[1] += time.time()-learn_start

        labels  = self._labels(actions, action, reward, probability)

        self._vw.learn(self._vw.make_examples({'x':self._flat(context)}, adfs, labels))

    def _labels(self,actions,action,reward:float,prob:float):
        return [ f"{i+1}:{round(-reward,5)}:{round(prob,5)}" if a == action else None for i,a in enumerate(actions)]

    def _flat(self,features:Any) -> Any:
        if features is None or isinstance(features,(int,float,str)):
            return features
        elif isinstance(features,dict):
            new_items = {}
            for k,v in features.items():
                if v is None or isinstance(v, (int,float,str)):
                    new_items[str(k)] = v
                else:
                    new_items.update( (f"{k}_{i}",f)  for i,f in enumerate(v))
            return new_items

        else:
            return [ff for f in features for ff in (f if isinstance(f,tuple) else [f]) ]

    def __reduce__(self):
        return (type(self), (self._epsilon, self._cmt, *self._args))