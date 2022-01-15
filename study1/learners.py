import time
import math

from statistics import mean

from typing import Hashable, Sequence, Dict, Any

from memory import CMT

from coba.contexts import LearnerContext
from coba.encodings import InteractionsEncoder

logn = 500
bits = 20

class MemoryKey:

    def __init__(self, context, action) -> None:

        self.context  = InteractionsEncoder(["x"]).encode(x=context)
        self.action   = InteractionsEncoder(["a"]).encode(a=action)

        self._hash = hash((context,action))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, MemoryKey) and self.context == __o.context and self.action == __o.action

class MemorizedLearner:

    def __init__(self, epsilon: float, cmt: CMT) -> None:

        assert 0 <= epsilon and epsilon <= 1

        self._epsilon = epsilon
        self._i       = 0
        self._cmt     = cmt
        self._times   = [0, 0]

    @property
    def params(self) -> Dict[str,Any]:
        return { 'family': 'memorized_taken','e':self._epsilon, **self._cmt.params }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        self._i += 1

        if logn and self._i % logn == 0:
           print(f"MEM {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"MEM {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        predict_start = time.time()

        paths,rewards = zip(*[self._cmt.query(MemoryKey(context, a)) or 0 for a in actions])

        LearnerContext.logger.write(avg_depths=mean([len(p) for p in paths]))

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
        
        return [ grd_p+min_p if a in greedy_A else min_p for a in actions ]

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, predict_info: Any) -> None:
        """Learn about the result of an action that was taken in a context."""

        learn_start = time.time()

        memory_key = MemoryKey(context, action)

        self._cmt.update(key=memory_key, outcome=reward)
        self._cmt.insert(key=memory_key, value=reward)

        self._times[1] += time.time()-learn_start
