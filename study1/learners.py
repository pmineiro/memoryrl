import time
import math
import statistics
import random

from typing_extensions import Literal
from typing import Hashable, Sequence, Dict, Any

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from coba.encodings import InteractionsEncoder

from memory import CMT

logn = 500
bits = 20

class MemoryKey:
    time = 0
    def __init__(self, context, action, interactions) -> None:

        features = self._featurize(context, action, interactions)

        if isinstance(features[0],tuple):
            self._features = FeatureHasher(n_features=2**bits, input_type='pair').fit_transform([features])
            self._features.sort_indices()
        else:
            self._features = np.array([features])

        self._hash = hash((context,action))

        self.context = context
        self.action  = action 

    def features(self):
        return self._features

    def _featurize(self, context, action, interactions):
        return InteractionsEncoder(interactions).encode(x=context,a=action)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, MemoryKey) and self.context == __o.context and self.action == __o.action

class OmegaDiffLearner:

    def __init__(self, epsilon: float, cmt: CMT, X = ["x","a","xa","xxa"], signal: Literal["abs","^2"] = "^2", megalr:float = 0.1, sort: bool = False) -> None:

        assert 0 <= epsilon and epsilon <= 1
        
        self._epsilon = epsilon
        self._i       = 0
        self._cmt     = cmt
        self._times   = [0, 0]
        self._signal  = signal
        self._X       = X
        self._megalr  = megalr
        self._sort    = sort

    @property
    def params(self) -> Dict[str,Any]:
        return { 'family': 'omega_diff', 'e':self._epsilon, **self._cmt.params, "X": self._X, 'ml': self._megalr, "sig": self._signal, 'srt':self._sort }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        self._i += 1

        if logn and self._i % logn == 0:
           print(f"MEM {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"MEM {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        predict_start = time.time()

        omegas = []
        depths = []
        counts = []

        for action in actions:
            trigger = MemoryKey(context, action, self._X)
            (_, Z, l) = self._cmt.query(trigger, 1, 0)

            omegas.append(Z[0][1] if Z else -math.inf)
            depths.append(l.depth)
            counts.append(len(l.memories))

        greedy_r = -math.inf
        greedy_A = []

        for action, mem_value in zip(actions, omegas):
            if mem_value == greedy_r:
                greedy_A.append(action)
            if mem_value > greedy_r:
                greedy_r = mem_value
                greedy_A = [action]

        minp = self._epsilon / len(actions)
        grdp = (1-self._epsilon)/len(greedy_A)

        self._times[0] += time.time()-predict_start

        info = {'action_leaf_depths':depths, 'action_leaf_sizes': counts, 'action_memories': omegas }
        return [ grdp+minp if a in greedy_A else minp for a in actions ], (info, actions)

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, predict_info: Any) -> None:
        """Learn about the result of an action that was taken in a context."""

        action_info, actions = predict_info 
        action_info["action_index"] = actions.index(action)

        learn_start = time.time()

        key = MemoryKey(context, action, self._X)

        if self._megalr > 0:
            (u,Z,l) = self._cmt.query(key, k=1, epsilon=0) #we pick best with prob 1-epsilon
            if len(Z) > 0:
                self._cmt.update_omega(Z[0][0], Z[0][1]+self._megalr*(reward-Z[0][1]))

        (u,Z,l) = self._cmt.query(key, k=2, epsilon=1) #we pick best with prob 1-epsilon

        if self._sort:
            Z = sorted(Z, key=lambda z: abs(reward-z[1]))

        # current ranking loss:
        # 1. using the top 2 results from the current scorer, order them "better"
        # alternative ranking loss:
        # 1. find the best result in the leaf (using the "extra" learn-only information)
        # 2. induce a ranking loss wrt the top result from the scorer
        # 3. special case: if top scorer result == best result in leaf, use second scorer result 
        #    note we esesntially are always doing the special case right now

        if len(Z) > 0:
            error = reward-Z[0][1]
            error = abs(error) if self._signal == "abs" else (error)**2
            self._cmt.update(u, key, Z, 1-error)
        
        if key not in self._cmt.leaf_by_mem_key:
            v = self._cmt.insert(key, reward)
        else:
            v = None

        update_info = {
            'update_mem_leaf_size'  : len(l.memories),
            'update_mem_leaf_depth' : l.depth,
            'insert_mem_leaf_size'  : len(v.memories)-1 if v else None,
            'insert_mem_leaf_depth' : v.depth if v else None
        }

        self._times[1] += time.time()-learn_start

        depths  = [ n.depth                                                                 for n in self._cmt.nodes if n.is_leaf ]
        cnts    = [ len(n.memories)                                                         for n in self._cmt.nodes if n.is_leaf ]
        avgs    = [ 0 if len(n.memories) == 0 else statistics.mean(n.memories.values())     for n in self._cmt.nodes if n.is_leaf ]
        vars    = [ 0 if len(n.memories) <= 1 else statistics.variance(n.memories.values()) for n in self._cmt.nodes if n.is_leaf ]

        tree_info = {"leaf_depths": depths, "leaf_mem_cnt": cnts, "leaf_mem_avg": avgs, "leaf_mem_var": vars }

        return {**action_info, **update_info, **tree_info}

class RewarDirectLearner:

    def __init__(self, epsilon: float, cmt: CMT, X = ["x","a","xa","xxa"], explore:Literal["each","every"]="each", megalr:float = 0.1,) -> None:

        assert 0 <= epsilon and epsilon <= 1
        
        self._epsilon = epsilon
        self._i       = 0
        self._cmt     = cmt
        self._times   = [0, 0]
        self._X       = X
        self._explore = explore
        self._rng     = random.Random(0xbeef)
        self._megalr  = megalr

    @property
    def params(self) -> Dict[str,Any]:
        return { 'family': 'reward_dir', 'e':self._epsilon, **self._cmt.params, "X": self._X, 'ml': self._megalr,"exp": self._explore}

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
        return [ int(a in greedy_A)/len(greedy_A) for a in actions ], (info, actions, updates)

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, predict_info: Any) -> None:
        """Learn about the result of an action that was taken in a context."""

        action_info, actions, updates = predict_info 
        action_info["action_index"] = actions.index(action)

        learn_start = time.time()

        key = MemoryKey(context, action, self._X)

        if self._megalr > 0:
            (u,Z,l) = self._cmt.query(key, k=1, epsilon=0) #we pick best with prob 1-epsilon
            if len(Z) > 0:
                self._cmt.update_omega(Z[0][0], Z[0][1]+self._megalr*(reward-Z[0][1]))

        for update in updates:
            if len(update[2]) > 0:
                self._cmt.update(*update, reward)

        if key not in self._cmt.leaf_by_mem_key:
            self._cmt.insert(key, reward)

        self._times[1] += time.time()-learn_start

        depths  = [ n.depth                                                                 for n in self._cmt.nodes if n.is_leaf ]
        cnts    = [ len(n.memories)                                                         for n in self._cmt.nodes if n.is_leaf ]
        avgs    = [ 0 if len(n.memories) == 0 else statistics.mean(n.memories.values())     for n in self._cmt.nodes if n.is_leaf ]
        vars    = [ 0 if len(n.memories) <= 1 else statistics.variance(n.memories.values()) for n in self._cmt.nodes if n.is_leaf ]

        tree_info = {"leaf_depths": depths, "leaf_mem_cnt": cnts, "leaf_mem_avg": avgs, "leaf_mem_var": vars }

        return {**action_info, **tree_info}

# We used to have a ResidualLearner too. The idea was that we'd learn a VW regressor 
# to predict context/action values and then use the memory tree to learn the residual 
# of the VW regressor predictions. 
# Our VW regressor was --quiet -b {bits} --cb_adf -q sa --cubic ssa --ignore_linear s
# I removed the learner because it was way way out of date and didn't seem to do well
# on a larger set of environments.
