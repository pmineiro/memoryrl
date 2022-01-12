import time
import math
import random

from typing_extensions import Literal
from typing import Hashable, Sequence, Dict, Any

from coba.encodings import InteractionsEncoder

from memory import CMT

logn = 500
bits = 20

class MemoryKey:
    time = 0
    def __init__(self, context, action, interactions) -> None:
        
        features = InteractionsEncoder(interactions).encode(x=context,a=action)

        if isinstance(features,dict):
            self.features = {(hash(k) % 2**bits):float(v) for k,v in features.items()}
        else:
            self.features = [ float(f) for f in features]

        self.context  = context
        self.action   = action 

        self._hash = hash((context,action))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, MemoryKey) and self.context == __o.context and self.action == __o.action

class MemorizedLearner:

    def __init__(self, 
        predict_eps  : float,
        learn_eps    : float,
        signal       : Literal["rwd","d^2","|d|"],
        cmt          : CMT,
        X            : Sequence[str] = ["x","a","xa","xxa"],
        explore      : Literal["each","every","e-greedy"] = "each") -> None:

        assert 0 <= predict_eps and predict_eps <= 1

        self._predict_eps = predict_eps
        self._learn_eps   = learn_eps
        self._i           = 0
        self._cmt         = cmt
        self._signal      = signal
        self._times       = [0, 0]
        self._X           = X
        self._explore     = explore
        self._rng         = random.Random(0xbeef)

    @property
    def params(self) -> Dict[str,Any]:
        return { 
            'family': 'memorized_taken',
            'p_e':self._predict_eps,
            'l_e':self._learn_eps,
            'sig': self._signal,
            **self._cmt.params,
            'X': self._X,
        }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        self._i += 1

        if logn and self._i % logn == 0:
           print(f"MEM {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"MEM {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        predict_start = time.time()

        if self._explore == "each":
            # we greed EACH action with prob 1-per_action_epsilon
            # this gives 1-self._epsilon chance of greeding all actions and
            # and self._epsilon chance of exploring on at least one action
            per_action_epsilon = 1-(1-self._predict_eps)**(1/len(actions))
        
        elif self._explore == "e-greedy":
            per_action_epsilon = 0

        else:
            # we greed EVERY action with prob 1-self._epsilon 
            # and explore EVERY action with prob self._epsilon
            per_action_epsilon = int(self._rng.uniform(0,1) <= self._predict_eps)

        omegas  = []
        depths  = []
        counts  = []
        updates = []

        for action in actions:
            query_key = MemoryKey(context, action, self._X)
            (u, Z, l) = self._cmt.query(query_key, 2, per_action_epsilon)

            updates.append([u, query_key, Z])
            omegas .append(Z[0][1] if Z else -math.inf)
            depths .append(l.depth)
            counts .append(len(l.memories))

        greedy_r = -math.inf
        greedy_A = []

        for action, mem_value in zip(actions, omegas):
            if mem_value == greedy_r:
                greedy_A.append(action)
            if mem_value > greedy_r:
                greedy_r = mem_value
                greedy_A = [action]

        self._times[0] += time.time()-predict_start

        if self._explore == 'e-greedy':
            minp = self._predict_eps / len(actions)
            grdp = (1-self._predict_eps)/len(greedy_A)
            probs = [ grdp+minp if a in greedy_A else minp for a in actions ]
        else:
            probs = [ int(a in greedy_A)/len(greedy_A) for a in actions ]

        return probs

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, predict_info: Any) -> None:
        """Learn about the result of an action that was taken in a context."""

        learn_start = time.time()

        query_key = MemoryKey(context, action, self._X)

        (u, Z, l) = self._cmt.query(query_key, 2, epsilon=self._learn_eps)
        
        if len(Z) > 0:
            if self._signal == 'rwd':
                rewards =  [ reward if i == 0 else 0 for i,z in enumerate(Z) ]
            elif self._signal == "|d|":
                rewards = [ 1-(reward-z[1]) for z in Z ]
            else:
                rewards = [ 1-(reward-z[1])**2 for z in Z ]

            self._cmt.update(u, query_key, rewards[0], Z, rewards)

        if query_key not in self._cmt.leaf_by_mem_key:
            self._cmt.insert(query_key, reward)

        self._times[1] += time.time()-learn_start