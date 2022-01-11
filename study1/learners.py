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
        epsilon      : float, 
        signal       : Literal["rwd","d^2","|d|"], 
        cmt          : CMT, 
        X            : Sequence[str] = ["x","a","xa","xxa"], 
        explore      : Literal["each","every","e-greedy"] = "each", 
        every_update : bool = True,
        taken_update : int = 0,
        megalr       : float = 0.1,
        sort         : bool = True,
        direct_update: bool = False) -> None:

        assert 0 <= epsilon and epsilon <= 1

        self._epsilon  = epsilon
        self._i        = 0
        self._cmt      = cmt
        self._signal   = signal
        self._times    = [0, 0]
        self._X        = X
        self._explore  = explore
        self._rng      = random.Random(0xbeef)
        self._megalr   = megalr
        self._E_update = every_update
        self._T_update = taken_update
        self._sort     = sort
        self._du       = direct_update

    @property
    def params(self) -> Dict[str,Any]:
        return { 
            'family': 'memorized_taken', 
            'e':self._epsilon, 
            'sig': self._signal, 
            **self._cmt.params, 
            'X': self._X, 
            'ml': self._megalr, 
            'du' : self._du
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
            per_action_epsilon = 1-(1-self._epsilon)**(1/len(actions))
        
        elif self._explore == "e-greedy":
            per_action_epsilon = 0
        
        else:
            # we greed EVERY action with prob 1-self._epsilon 
            # and explore EVERY action with prob self._epsilon
            per_action_epsilon = int(self._rng.uniform(0,1) <= self._epsilon)

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

        info = {'action_leaf_depths':depths, 'action_leaf_sizes': counts, 'action_memories': omegas }

        if self._explore == 'e-greedy':
            minp = self._epsilon / len(actions)
            grdp = (1-self._epsilon)/len(greedy_A)
            probs = [ grdp+minp if a in greedy_A else minp for a in actions ]
        else:
            probs = [ int(a in greedy_A)/len(greedy_A) for a in actions ]

        return probs, (info, actions, updates, omegas)

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, predict_info: Any) -> None:
        """Learn about the result of an action that was taken in a context."""

        action_info, actions, updates,omegas = predict_info 
        action_info["action_index"] = actions.index(action)

        learn_start = time.time()

        key = MemoryKey(context, action, self._X)

        if self._megalr > 0:
            (u,Z,l) = self._cmt.query(key, k=1, epsilon=0) #we pick best with prob 1-epsilon
            if len(Z) > 0:
                self._cmt.update_omega(Z[0][0], Z[0][1]+self._megalr*(reward-Z[0][1]))

        omega = omegas[actions.index(action)]
        loss  = 1-reward if self._signal == 'rwd' else abs(reward-omega) if self._signal == "|d|" else  (reward-omega)**2

        old_t = self._cmt.f.t 
        old_m = len(self._cmt.leaf_by_mem_key)

        if self._E_update:
            for update in updates:
                if len(update[2]) > 0:
                    self._cmt.update(*update, 1-loss)
                    if self._du:
                        self._cmt.f.update(update[1], [z[0] for z in update[2]], 1-loss)

        for _ in range(self._T_update):
            (u, Z, l) = self._cmt.query(key, 2, 1)
            if len(Z) > 0:
                omega = Z[0][1]
                loss  = 1-reward if self._signal == 'rwd' else abs(reward-omega) if self._signal == "|d|" else  (reward-omega)**2

                # current ranking loss:
                # 1. using the top 2 results from the current scorer, order them "better"
                # alternative ranking loss:
                # 1. find the best result in the leaf (using the "extra" learn-only information)
                # 2. induce a ranking loss wrt the top result from the scorer
                # 3. special case: if top scorer result == best result in leaf, use second scorer result
                #    note we esesntially are always doing the special case right now

                if self._sort:
                    Z = sorted(Z, key=lambda z: abs(reward-z[1]))

                self._cmt.update(u, key, Z, 1-loss)

        if key not in self._cmt.leaf_by_mem_key:
            self._cmt.insert(key, reward)

        # if self._cmt.f.t > old_t:
        #     print(f'update on {self._i}')

        # if old_m < len(self._cmt.leaf_by_mem_key):
        #     print(f"split on {self._i}")

        self._times[1] += time.time()-learn_start

# We used to have a ResidualLearner too. The idea was that we'd learn a VW regressor 
# to predict context/action values and then use the memory tree to learn the residual 
# of the VW regressor predictions. 
# Our VW regressor was --quiet -b {bits} --cb_adf -q sa --cubic ssa --ignore_linear s
# I removed the learner because it was way way out of date and didn't seem to do well
# on a larger set of environments.
