import time
import random
import math
import copy

from typing import Hashable, Sequence, Dict, Any, Optional, Tuple, Union, List

import numpy as np
from vowpalwabbit import pyvw
from sklearn.feature_extraction import FeatureHasher

from coba.encodings import InteractionTermsEncoder
from coba.learners import VowpalLearner, Learner, CorralLearner

from memory import CMT
from scorers import ClassScorer
from feedbacks import DevFeedback
from examplers import IdentityExampler, PureExampler

logn = 500
bits = 20

class CMT_Implemented:

    class MemoryKey:
        time = 0
        def __init__(self, context, action, interactions) -> None:

            features = self._featurize(context, action, interactions)

            if isinstance(features[0],tuple):
                self._features = FeatureHasher(n_features=2**16, input_type='pair').fit_transform([features])
                self._features.sort_indices()
            else:
                #doing this because some code later that assumes we're working with sparse matrices
                #self._features = sp.csr_matrix((features,([0]*len(features),range(len(features)))), shape=(1,len(features)))
                self._features = np.array([features])

            self._hash = hash((context,action))

            self.context = context
            self.action  = action 

        def features(self):
            return self._features

        def _featurize(self, context, action, interactions):
            return InteractionTermsEncoder(interactions).encode(x=context,a=action)

        def __hash__(self) -> int:
            return self._hash

    class LogisticModel_VW:
        def __init__(self, *args, **kwargs):
            self.vw = pyvw.vw(f'--quiet -b {bits} --loss_function logistic --noconstant --power_t 1 --link=glf1')
            self.exampler = IdentityExampler()

        def predict(self, xraw):
            return self.vw.predict(self.exampler.make_example(self.vw, xraw.features()))

        def update(self, xraw, y, w):
            self.vw.learn(self.exampler.make_example(self.vw, xraw.features(), 0, y, w))

        def __reduce__(self):
            return (CMT_Implemented.LogisticModel_VW,())

    class LogisticModel_SK:
        def __init__(self):

            from sklearn.linear_model import SGDClassifier

            self.clf  = SGDClassifier(loss="log", average=True, learning_rate='constant', eta0=0.5)
            self.is_fit = False
            self.time = 0

        def predict(self, x):
            return 1 if not self.is_fit else self.clf.predict(self._domain(x))[0]

        def update(self, x, y, w):
            start = time.time()
            self.clf.partial_fit(self._domain(x), [y], sample_weight=[w], classes=[-1,1])
            self.is_fit = True
            self.time += time.time()-start

        def _domain(self, x):
            return x.features()

    def __init__(self, max_memories: int = 1000, router_type:str = 'sk', scorer=ClassScorer(), feedback=DevFeedback(), c=10, d=1, megalr=0.1, interactions=["x","a","xa","xxa"], g: float = 0, sort:bool = False, alpha:float=0.25) -> None:

        assert 1 <= max_memories

        self._max_memories = max_memories
        self._router_type  = router_type
        self._c            = c
        self._d            = d
        self._megalr       = megalr
        self._interactions = interactions
        self._gate         = g
        self._sort         = sort
        self._alpha        = alpha

        router_factory = CMT_Implemented.LogisticModel_SK if self._router_type == 'sk' else CMT_Implemented.LogisticModel_VW

        self._scorer = copy.deepcopy(scorer)
        self._signal = feedback

        random_state   = random.Random(1337)
        ords           = random.Random(2112)

        self.mem = CMT(router_factory, self._scorer, alpha=self._alpha, c=self._c, d=self._d, randomState=random_state, optimizedDeleteRandomState=ords, maxMemories=self._max_memories)

    @property
    def params(self):
        return { 'm': self._max_memories, 'd': self._d, 'c': self._c, 'ml': self._megalr, "X": self._interactions, "g": self._gate, "a": self._alpha, "srt": self._sort, "sig": self._signal.params, "rt": self._router_type, "scr": self._scorer.params }

    def query(self, context: Hashable, actions: Sequence[Hashable], default = None, topk:int=1):

        memories = []

        for action in actions:
            trigger = self._mem_key(context, action)
            (_, z) = self.mem.query(trigger, topk, 0)
            memories.extend([ zz[1] for zz in z ] or [default])

        return memories

    def update(self, context, action, observation, reward):

        trigger = self._mem_key(context, action)
        
        (u,z) = self.mem.query(trigger, k=2, epsilon=1) #we pick best with prob 1-epsilon

        if self._sort:
            z = sorted(z, key=lambda zz: abs(observation-zz[1]))

        # current ranking loss:
        # 1. using the top 2 results from the current scorer, order them "better"
        # alternative ranking loss:
        # 1. find the best result in the leaf (using the "extra" learn-only information)
        # 2. induce a ranking loss wrt the top result from the scorer
        # 3. special case: if top scorer result == best result in leaf, use second scorer result 
        #    note we esesntially are always doing the special case right now

        if len(z) > 0:

            if self._megalr > 0:
                z[0] = (z[0][0], z[0][1]+self._megalr*(observation-z[0][1]))
                self.mem.updateomega(z[0][0], z[0][1])

            signal = self._signal.signal(observation, z[0][1], reward)
            self.mem.update(u, trigger, z, signal)
        else:
            signal = 0

        no_memory  = len(z) == 0
        not_maxed  = len(self.mem.leafbykey) < self.mem.maxMemories
        diff_mem   = len(z) > 0 and abs(observation-z[0][1]) >= self._gate
        is_new_trg = trigger not in self.mem.leafbykey

        if is_new_trg and not (no_memory or not_maxed or diff_mem):
            pass
            #print("!")

        if is_new_trg and (no_memory or not_maxed or diff_mem):
            self.mem.insert(trigger, observation)

    def _mem_key(self, context, action):
        return CMT_Implemented.MemoryKey(context,action,self._interactions)

class MemorizedLearner:

    def __init__(self, epsilon: float, mem: CMT_Implemented) -> None:

        assert 0 <= epsilon and epsilon <= 1
        
        self._epsilon = epsilon
        self._i       = 0
        self._mem     = mem
        self._times   = [0, 0]

    @property
    def family(self) -> str:
        return "CMT_Memorized"

    @property
    def params(self) -> Dict[str,Any]:
        return { 'e':self._epsilon, **self._mem.params }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        self._i += 1

        if logn and self._i % logn == 0:
           print(f"MEM {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"MEM {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        predict_start = time.time()

        greedy_as = []
        greedy_r = -math.inf

        for action, remembered_value in zip(actions, self._mem.query(context, actions, default=-math.inf)):
            if remembered_value == greedy_r:
                greedy_as.append(action)
            if remembered_value > greedy_r: 
                greedy_r = remembered_value
                greedy_as = [action]

        minp = self._epsilon / len(actions)
        grdp = (1-self._epsilon)/len(greedy_as)

        self._times[0] += time.time()-predict_start

        return [ grdp+minp if a in greedy_as else minp for a in actions ]

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, info: Any) -> None:
        """Learn about the result of an action that was taken in a context."""

        learn_start = time.time()
        self._mem.update(context, action, reward, reward)
        self._times[1] += time.time()-learn_start

class ResidualLearner:

    class ResidualSignal:
        def __init__(self,signal):
            self._signal = signal

        def signal(self, observed, memory, reward):
            return self._signal.signal(observed/2, memory/2, reward)

    def __init__(self, epsilon: float, mem: CMT_Implemented):

        assert 0 <= epsilon and epsilon <= 1

        from vowpalwabbit import pyvw

        self._args = (epsilon, mem)
        self._epsilon = epsilon

        self._mem = mem 
        self._vw  = pyvw.vw(f'--quiet -b {bits} --cb_adf -q sa --cubic ssa --ignore_linear s')

        self._mem._signal = ResidualLearner.ResidualSignal(self._mem._signal)

        self._i        = 0
        self._times    = [0,0]

    @property
    def family(self) -> str:
        return "CMT_Residual"

    @property
    def params(self) -> Dict[str,Any]:
        return  { 'e':self._epsilon,  **self._mem.params }

    def toadf(self, context, actions, label=None):

        if len(context) == 2 and isinstance(context[0], tuple) and isinstance(context[1], tuple):
            context = list(zip(*context))
        else:
            context = list(enumerate(context))

        return '\n'.join([
            'shared |s ' + ' '.join([ f'{k+1}:{v}' for k, v in context ]),
        ] + [
            f'{dacost} |a ' + ' '.join([ f'{k+1}:{v}' for k, v in enumerate(a) if v != 0 ])
            for n, a in enumerate(actions)
            for dacost in ((f'0:{label[1]}:{label[2]}' if label is not None and n == label[0] else ''),)
        ])

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Tuple[Sequence[float], Any]:
        """Choose which action index to take."""

        predict_start = time.time()
        self._i += 1

        if logn and self._i % logn == 0:
           print(f"RES {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"RES {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        predicted_losses = self._vw.predict(self.toadf(context, actions))
        rememberd_deltas = self._mem.query(context, actions, 0)

        greedy_as = []
        greedy_l = math.inf

        for action, remembered_value in zip(actions, map(sum,zip(predicted_losses, rememberd_deltas))):
            if remembered_value == greedy_l:
                greedy_as.append(action)
            if remembered_value < greedy_l: 
                greedy_l  = remembered_value
                greedy_as = [action]

        minp = self._epsilon / len(actions)
        grdp = (1-self._epsilon)/len(greedy_as)

        self._times[0] += time.time()-predict_start

        prediction = [ grdp+minp if a in greedy_as else minp for a in actions ]
        info       = (predicted_losses, actions)

        return (prediction,info)

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, info: Any) -> None:
        """Learn about the result of an action that was taken in a context."""

        learn_start = time.time()

        predicted_losses = info[0]
        actions          = info[1]

        act_ind  = actions.index(action)
        prd_loss = predicted_losses[act_ind]

        obs_loss  = 1-reward
        obs_resid = obs_loss-prd_loss

        self._vw.learn(self.toadf(context, actions, (act_ind, obs_loss, probability)))
        self._mem.update(context, action, obs_resid, reward)

        self._times[1] += time.time()-learn_start

    def __reduce__(self):
        return (ResidualLearner,self._args)

class MemCorralLearner(CorralLearner):
    """This is a modified implementation of the Agarwal et al. (2017) Corral algorithm.

    This algorithm assumes that the reward distribution has support in [0,1] and that all learners can learn off-policy.

    References:
        Agarwal, Alekh, Haipeng Luo, Behnam Neyshabur, and Robert E. Schapire. 
        "Corralling a band of bandit algorithms." In Conference on Learning 
        Theory, pp. 12-38. PMLR, 2017.
    """

    id = 0

    def __init__(self, base_learners: Sequence[Learner], eta : float, T: float = math.inf, type: str = "off-policy", seed: int = 1) -> None:
        """Instantiate a CorralLearner.

        Args:
            base_learners: The collection of algorithms to use as base learners.
            eta: The learning rate. In our experiments a value between 0.05 and .10 often seemed best.
            T: The number of interactions expected during the learning process. In our experiments the 
                algorithm performance seemed relatively insensitive to this value.
            seed: A seed for a random number generation in ordre to get repeatable results.

        Remark:
            If the given base learners don't require feedback for every selected action then it is possible to
            modify this algorithm so as to use rejection sampling while being as efficient as importance sampling.
        """

        self._full_expected_rwd = [0]*len(base_learners)
        self._i = 0
        self._id = MemCorralLearner.id
        MemCorralLearner.id += 1

        super().__init__(base_learners, eta=eta, T=T, seed=seed, type=type)

    def predict(self, context, actions) -> Tuple[Sequence[float], Any]:

        self._i += 1

        return super().predict(context, actions)

    def learn(self, context, action, reward, probability, info) -> Dict[str,Any]:

        base_preds = info[3]
        actions    = info[4]

        picked_index = actions.index(action)
        instant_rwd  = [ reward * base_pred[picked_index]/probability for base_pred in base_preds ]

        for i,r in enumerate(instant_rwd):
            self._full_expected_rwd[i] = (1-1/self._i) * self._full_expected_rwd[i] + (1/self._i) * r  

        # if self._i % int(logn/5) == 0:
        #     print(self._id)
        #     print(self._p_bars)
        #     print(self._full_expected_rwd)
        #     print("")

        return super().learn(context, action, reward, probability, info)

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.

        See the base class for more information
        """

        def base_name(i):
            if i == 0:
                return self._base_learners[i].family
            else:
                return f"{self._base_learners[i].family}({self._base_learners[i].params})"
            
        return { "type":self._type, "B": [ base_name(i) for i in range(len(self._base_learners)) ] }


class MemorizedVW:

    def __init__(self, epsilon: float) -> None:
        assert 0 <= epsilon and epsilon <= 1
        self._vw = VowpalLearner(f"--cb_explore_adf --epsilon {epsilon} --random_seed 1 --cb_type dm --interactions ssa --interactions sa --ignore_linear s")
        self._epsilon = epsilon

    @property
    def family(self) -> str:
        return "VW_Memorized"

    @property
    def params(self) -> Dict[str,Any]:
        return { 'e':self._epsilon }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        return self._vw.predict(context, actions)

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, info: Any) -> None:
        """Learn about the result of an action that was taken in a context."""
        
        self._vw.learn(context, action, reward, 1, info)
