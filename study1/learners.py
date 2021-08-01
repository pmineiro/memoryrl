import time
import random
import math

from itertools import count
from collections import defaultdict
from typing import Hashable, Sequence, Dict, Any, Optional, Tuple, Union, List

import numpy as np
from sklearn.feature_extraction import FeatureHasher

from coba.learners import VowpalLearner, Learner

from memory import CMT

logn = 500
bits = 20

class CMT_Implemented:

    class MemoryKey:

        def __init__(self, context, action) -> None:

            context_features = self._featurize(context, 2, 'x')
            action_features  = self._featurize(action , 1, 'a')

            x   = context_features[1]
            xx  = context_features[2]
            a   = action_features[1]
            xa  = [ (x_f[0]+a_f[0],x_f[1]*a_f[1]) for x_f in x  for a_f in a ]
            xxa = [ (x_f[0]+a_f[0],x_f[1]*a_f[1]) for x_f in xx for a_f in a ]

            all_features = x+a+xa+xxa
            self._hashed = FeatureHasher(n_features=2**18, input_type='pair').fit_transform([all_features])

            #if len(all_features) != self._hashed.nnz:
                #print(self._hashed.nnz/len(all_features))

            self._hash = hash(tuple(x+a))

        def features(self):
            return self._hashed

        def _featurize(self, input, degree, ns):

            features_by_degree = defaultdict(list)

            if input is None:
                features_by_degree[1] = [ (ns,1) ]

            elif isinstance(input,str):
                features_by_degree[1] = [ (ns+input, 1) ]

            else:
                if isinstance(input,dict):
                    items = input.items()
                else:
                    items = list(enumerate(input))

                items = [ (f"{ns}{n}{v}",1) if isinstance(v,str) else (f"{ns}{n}",v) for n,v in items ]
                items = sorted(filter(lambda i: i[1] !=0, items), key=lambda i: i[0])
                names,values = zip(*items)

                features_by_degree[1] = list(zip(names,values))

                if degree == 2:
                    features_by_degree[2] = [ (names[i]+names[j], values[i]*values[j]) for i in range(len(names)) for j in range(i+1) ]

            return features_by_degree

        def __hash__(self) -> int:
            return self._hash

    class LogisticModel_VW:
        def __init__(self, *args, **kwargs):

            from vowpalwabbit import pyvw
            self.vw = pyvw.vw(f'--quiet -b {bits} --loss_function logistic --link=glf1 -q ax --cubic axx')

        def predict(self, xraw):

            (x, a) = xraw
            ex = ' |x ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(x) if v != 0]
            )  + ' |a ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(a) if v != 0]
            )

            return self.vw.predict(ex)

        def update(self, xraw, y, w):

            (x, a) = xraw
            assert y == 1 or y == -1
            assert w >= 0
            ex = f'{y} {w} |x ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(x) if v != 0]
            )  + ' |a ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(a) if v != 0]
            )

            self.vw.learn(ex)

    class LogisticModel_SK:
        def __init__(self):

            from sklearn.linear_model import SGDClassifier

            self.clf  = SGDClassifier(loss="log", average=True)
            self.is_fit = False

        def predict(self, x): 
            return 1 if not self.is_fit else self.clf.predict(self._domain(x))[0]

        def update(self, x, y, w):
            self.is_fit = True
            self.clf.partial_fit(self._domain(x), [y], sample_weight=[w], classes=[-1,1])

        def _domain(self, x):
            return x.features()

    class LearnedEuclideanDistance:

        def __init__(self,learn=True):

            from vowpalwabbit import pyvw

            self.vw    = None if not learn else pyvw.vw(f'--quiet -b {bits} --noconstant --loss_function logistic --link=glf1')            
            self.tt    = [0]*6
            self.learn = learn

        def diff(self, vec1, vec2):            
            return vec1-vec2

        def inner(self, vec1, vec2) -> float:
            return vec1.multiply(vec2).data.sum()

        def predict(self, xraw, z):

            ss = [0]*6
            ee = [0]*6

            X  = xraw.features() #now has (x,a,xa,xxa)... the old version just had (x,a,xa)
            Xp = z[0].features() #now has (x,a,xa,xxa)... the old version just had (x,a,xa)

            ss[2] = time.time()
            dp = self.diff(X,Xp)
            ee[2] = time.time()

            ss[3] = time.time()
            dp_dot_dp = self.inner(dp,dp) 
            ee[3] = time.time()

            ss[4] = time.time()
            initial    = -0.01 * dp_dot_dp
            features   = [f'{k}:{round(v**2,6)}' for k,v in zip(dp.indices, dp.data)]
            prediction = 0 if not self.vw else self.vw.predict(f' |x ' + ' '.join(features))
            ee[4] = time.time()

            for i,s,e in zip(count(),ss,ee):
                self.tt[i] += e-s

            return initial + prediction

        def update(self, xraw, z, r):
            if not self.vw: return

            if r > 0 and len(z) > 1:
                X   = xraw.features()    #now has (x,a,xa,xxa)... the old version just had (x,a,xa)
                Xp  = z[0][0].features() #now has (x,a,xa,xxa)... the old version just had (x,a,xa)
                Xpp = z[1][0].features() #now has (x,a,xa,xxa)... the old version just had (x,a,xa)

                dp  = self.diff(X, Xp)
                dpp = self.diff(X, Xpp)

                initial  = 0.01 * (self.inner(dp,dp) - self.inner(dpp,dpp))
                keys     = set(dp.indices) | set(dpp.indices)
                features = [f'{key}:{round(dp[0,key]**2-dpp[0,key]**2,6)}' for key in keys]

                self.vw.learn(f'1 {r} {initial} |x ' + ' '.join(features))

    def __init__(self, max_memories: int = 1000, learn_dist: bool = True, signal_type:str = 'se', router_type:str = 'sk', c=10, d=1, megalr=0.1) -> None:

        from sklearn.preprocessing import PolynomialFeatures

        self._max_memories = max_memories
        self._learn_dist   = learn_dist
        self._signal_type  = signal_type
        self._router_type  = router_type
        self._c            = c
        self._d            = d
        self._megalr       = megalr

        router_factory = CMT_Implemented.LogisticModel_SK if self._router_type == 'sk' else CMT_Implemented.LogisticModel_VW

        scorer         = CMT_Implemented.LearnedEuclideanDistance(self._learn_dist)
        random_state   = random.Random(1337)
        ords           = random.Random(2112)

        self._p = PolynomialFeatures(degree=2)

        self.mem = CMT(router_factory, scorer, alpha=0.25, c=self._c, d=self._d, randomState=random_state, optimizedDeleteRandomState=ords, maxMemories=self._max_memories)

    @property
    def params(self):
        return { 'm': self._max_memories, 'b': bits, 'ld': self._learn_dist, 'sig': self._signal_type, 'rt': self._router_type, 'd': self._d, 'c': self._c, 'mlr': self._megalr }

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return(CMT_Implemented, (self._max_memories, self._learn_dist, self._signal_type, self._router_type, self._c, self._d) )

    def query(self, context: Hashable, actions: Sequence[Hashable], default = None, topk:int=1):

        memories = []

        for action in actions:
            trigger = self._mem_key(context, action)
            (_, z) = self.mem.query(trigger, topk, 0)
            memories.extend([ zz[1] for zz in z ] or [default])

        return memories

    def update(self, context: Hashable, action: Hashable, memory: Union[float, np.ndarray]):

        trigger = self._mem_key(context, action)
        (u,z) = self.mem.query(trigger, k=2, epsilon=1)

        if len(z) > 0:
            updated_memory = z[0][1] + self._megalr * ( memory-z[0][1] )

            self.mem.updateomega(z[0][0], updated_memory)
            self.mem.update(u, trigger, z, self._error_signal(memory, updated_memory))

        if trigger in self.mem.leafbykey:
            self.mem.delete(trigger)

        self.mem.insert(trigger, memory)

    def _error_signal(self, obs, prd):

        if self._signal_type == 'se':
            if isinstance(prd, np.ndarray):
                return 1-min(1, max(0, (np.linalg.norm(prd-obs)**2)/2))
            else:
                return 1-(prd-obs)**2

        if self._signal_type == 're':

            how_much  = sorted([0, abs(obs)    , 1])[1]
            how_close = sorted([0, abs(obs-prd), 1])[1]

            return max(0, how_much - how_close)

        raise Exception(f"Unrecognized signal type: {self._signal_type}")

    def _mem_key(self, context, action):
        #start = time.time()
        a = CMT_Implemented.MemoryKey(context,action)
        #print(time.time()-start)
        return a

class FullFeedbackMemLearner:
    def __init__(self, max_memories: int = 1000, learn_dist: bool = True, c=10, d=1, signal:str = 'se', router: str = 'sk'):
        self._i       = 0

        self.mem = CMT_Implemented(max_memories, learn_dist, signal, router, c=c, d=d)

        self._full_reward = None
        self._fixed_actions = None

        self._times = [0, 0]

    @property
    def family(self) -> str:
        return "CMT_FullFeedback"

    @property
    def params(self) -> Dict[str,Any]:
        return self.mem.params

    def predict(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        predict_start = time.time()

        self._fixed_actions = self._fixed_actions or actions
        self._i += 1

        default_action = np.array([1] + [0]*(len(self._fixed_actions)-1))
        action_one_hot = self.mem.query(context, [()], default=default_action)[0]

        self._times[0] += time.time()-predict_start

        if self._full_reward is None:
            import coba.simulations
            import simulations
            
            if len(self._fixed_actions) == 3:
                self._full_reward = simulations.MemorizableSimulation().read().reward

            if len(self._fixed_actions) == 26:
                self._full_reward = coba.simulations.OpenmlSimulation(6).read().reward

            if len(self._fixed_actions) == 2:
                self._full_reward = coba.simulations.OpenmlSimulation(1471).read().reward

            if len(self._fixed_actions) == 1000:
                self._full_reward = coba.simulations.OpenmlSimulation(1592).read().reward

            if len(self._fixed_actions) == 51:
                self._full_reward = simulations.Rcv1Simulation().read().reward

            if len(self._fixed_actions) == 105:
                self._full_reward = simulations.SectorSimulation().read().reward

            if self._full_reward is None:
                raise Exception("We were unable to associate a simulation with the observed action set.")

        ga   = actions.index(self._fixed_actions[action_one_hot.argmax()])

        if logn and self._i % logn == 0:
           print(f"{self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"{self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        return [ 0 if i != ga else 1 for i in range(len(self._fixed_actions)) ]

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""        

        learn_start = time.time()

        true_one_hot = np.array(self._full_reward.observe([(key,context,a) for a in self._fixed_actions]))

        assert sum(true_one_hot) == 1, "Something is wrong with full feedback."

        self.mem.update(context, (), true_one_hot)

        self._times[1] += time.time()-learn_start        

class FullFeedbackVowpalLearner:
    def __init__(self):
        self._i = 0

        self._full_reward   = None
        self._fixed_actions = None

        self._times = [0, 0]

        self._vw = None

    @property
    def family(self) -> str:
        return "VW_FullFeedback"

    @property
    def params(self) -> Dict[str,Any]:
        return { }

    def predict(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        if self._vw is None:
            from vowpalwabbit import pyvw
            self._vw = pyvw.vw(f"--quiet --oaa {len(actions)} --interactions xx -b {bits}")

        predict_start = time.time()

        self._fixed_actions = self._fixed_actions or actions
        self._i += 1

        if self._full_reward is None:
            import coba.simulations
            import simulations

            if len(self._fixed_actions) == 3:
                self._full_reward = simulations.MemorizableSimulation().read().reward

            if len(self._fixed_actions) == 26:
                self._full_reward = coba.simulations.OpenmlSimulation(6).read().reward

            if len(self._fixed_actions) == 2:
                self._full_reward = coba.simulations.OpenmlSimulation(1471).read().reward

            if len(self._fixed_actions) == 1000:
                self._full_reward = coba.simulations.OpenmlSimulation(1592).read().reward

            if len(self._fixed_actions) == 51:
                self._full_reward = simulations.Rcv1Simulation().read().reward

            if len(self._fixed_actions) == 105:
                self._full_reward = simulations.SectorSimulation().read().reward

            if self._full_reward is None:
                raise Exception("We were unable to associate a simulation with the observed action set.")

        ga = actions.index(self._fixed_actions[self._vw.predict(f"|x {self._vw_features(context)}")-1])

        self._times[0] += time.time()-predict_start

        if logn and self._i % logn == 0:
           print(f"{self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"{self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        return [ 0 if i != ga else 1 for i in range(len(self._fixed_actions)) ]

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""        

        learn_start = time.time()

        true_one_hot = np.array(self._full_reward.observe([(key,context,a) for a in self._fixed_actions]))

        assert sum(true_one_hot) == 1, "Something is wrong with full feedback."

        self._vw.learn(f"{true_one_hot.argmax()+1} |x {self._vw_features(context)}")

        self._times[1] += time.time()-learn_start

    def _vw_features(self, context) -> str:
        if len(context) == 2 and isinstance(context[0],tuple) and isinstance(context[1],tuple):
            return " ".join([f"{k}:{v}" for k,v in zip(*context) ])
        else:
            return " ".join([f"{v}" for v in context])

class MemorizedIpsLearner:
    def __init__(self, epsilon:float, max_memories: int = 1000, learn_dist: bool = True, c=10, d=1, signal:str = 'se', router: str = 'sk', topk:int=10):
        self._i = 0

        self.mem = CMT_Implemented(max_memories, learn_dist, signal, router, c=c, d=d, megalr=0)

        assert c >= 40, "We take top 40 so increase c"

        self._epsilon = epsilon
        self._topk = topk
        self._fixed_actions = None

        self._times = [0, 0]

    @property
    def family(self) -> str:
        return "CMT_MemorizedIPS"

    @property
    def params(self) -> Dict[str,Any]:
        return {'e': self._epsilon, **self.mem.params, 'k': self._topk }

    def predict(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        predict_start = time.time()

        self._fixed_actions = self._fixed_actions or actions
        self._i += 1

        default_action = np.array([0]*len(self._fixed_actions))
        action_one_hot = np.stack(self.mem.query(context, [()], topk=self._topk, default=default_action)).mean(axis=0)

        max_indexes = np.argwhere(action_one_hot==action_one_hot.max()).flatten().tolist()

        ga   = [actions.index(self._fixed_actions[mi]) for mi in max_indexes ]
        minp = self._epsilon / len(actions)

        self._times[0] += time.time()-predict_start

        if logn and self._i % logn == 0:
           print(f"IPS {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"IPS {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        probs = [ minp if i not in ga else minp+(1-self._epsilon)/len(ga) for i in range(len(actions)) ]

        assert round(sum(probs),2) == 1, "ERROR"

        return probs

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""        

        learn_start = time.time()        

        self.mem.update(context, (), np.array([int(action==a)*reward/probability for a in self._fixed_actions]))

        self._times[1] += time.time()-learn_start        

class MemorizedLearner:

    def __init__(self, epsilon: float, max_memories: int = 1000, learn_dist: bool = True, c=10, d=1, signal:str = 'se', router: str = 'sk') -> None:

        assert 0 <= epsilon and epsilon <= 1
        assert 1 <= max_memories

        self._epsilon = epsilon
        self._i       = 0

        self.mem = CMT_Implemented(max_memories, learn_dist, signal, router, c=c, d=d)

        self._times = [0, 0]

    @property
    def family(self) -> str:
        return "CMT_Memorized"

    @property
    def params(self) -> Dict[str,Any]:
        return { 'e':self._epsilon,  **self.mem.params }

    def predict(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        predict_start = time.time()
        self._i += 1

        (greedy_a, greedy_r) = actions[0], -math.inf

        for action, remembered_value in zip(actions, self.mem.query(context, actions)):
            if remembered_value is None: continue
            if remembered_value > greedy_r: (greedy_a, greedy_r) = (action, remembered_value)

        ga   = actions.index(greedy_a)
        minp = self._epsilon / len(actions)

        self._times[0] += time.time()-predict_start

        if logn and self._i % logn == 0:
           print(f"MEM {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"MEM {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        return [ minp if i != ga else minp+(1-self._epsilon) for i in range(len(actions)) ]

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""
        learn_start = time.time()
        self.mem.update(context, action, reward)
        self._times[1] += time.time()-learn_start

class ResidualLearner:
    def __init__(self, epsilon: float, max_memories: int, learn_dist: bool = True, c=10, d=1, signal:str = 're', router:str ='sk'):

        assert 0 <= epsilon and epsilon <= 1
        assert 1 <= max_memories

        from vowpalwabbit import pyvw

        self._epsilon = epsilon
        self._max_memories = max_memories
        self._learn_dist = learn_dist
        self._c = c
        self._d = d
        self._signal = signal
        self._router = router
        
        self.mem = CMT_Implemented(max_memories, learn_dist, signal, router, c=c, d=d)
        self.vw  = pyvw.vw(f'--quiet -b {bits} --cb_adf -q sa --cubic ssa --ignore_linear s')

        self._i        = 0
        self._predicts = {}
        self._times    = [0,0]

    def __reduce__(self):
        return (ResidualLearner,(self._epsilon, self._max_memories, self._learn_dist, self._c, self._d, self._signal, self._router))

    @property
    def family(self) -> str:
        return "CMT_Residual"

    @property
    def params(self) -> Dict[str,Any]:
        return  { 'e':self._epsilon,  **self.mem.params }

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

    def predict(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        predict_start = time.time()
        self._i += 1

        predicts = self.vw.predict(self.toadf(context, actions))
        deltas   = self.mem.query(context, actions, 0)

        ga   = min(((p + dp, n) for p, dp, n in zip(predicts, deltas, range(len(actions)))))[1]
        minp = self._epsilon / len(actions)

        self._predicts[key] = (predicts, actions)

        self._times[0] += time.time()-predict_start

        if logn and self._i % logn == 0:
           print(f"RES {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"RES {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        return [ minp if i != ga else minp+(1-self._epsilon) for i in range(len(actions)) ]

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""

        learn_start = time.time()

        (predicts, actions) = self._predicts.pop(key)

        act_ind = actions.index(action)
        prd_loss = predicts[act_ind]

        obs_loss  = -reward
        obs_resid = obs_loss-prd_loss

        self.vw.learn(self.toadf(context, actions, (act_ind, obs_loss, probability)))
        self.mem.update(context, action, obs_resid)

        self._times[1] += time.time()-learn_start

class CorralRejection:
    """This is modified implementation of the Agarwal et al. (2017) Corral algorithm.

    This algorithm assumes that the reward distribution has support in [0,1].
    This algorithm also implements the remark on pg. 8 in order to improve 
    learning efficiency when multiple bandits select the same action.

    References:
        Agarwal, Alekh, Haipeng Luo, Behnam Neyshabur, and Robert E. Schapire. 
        "Corralling a band of bandit algorithms." In Conference on Learning 
        Theory, pp. 12-38. PMLR, 2017.
    """

    def __init__(self,
        max_memories: int,
        epsilon: float,
        eta : float,
        fix_count: int = None,
        T: float = math.inf,
        seed: int = None) -> None:
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

        learn_distance = True

        self._base_learners = [
            MemorizedLearner(epsilon, max_memories, learn_distance),
            ResidualLearner(epsilon, max_memories, learn_distance),
            VowpalLearner(epsilon=epsilon,seed=seed)
        ]

        M = 3

        self._gamma = 1/T
        self._beta  = 1/math.exp(1/math.log(T))

        self._eta_init = eta
        self._etas     = [ eta ] * M
        self._rhos     = [ float(2*M) ] * M
        self._ps       = [ 1/M ] * M
        self._p_bars   = [ 1/M ] * M

        self._fix_count = fix_count
        self._learn_count = 0

        self._random = random.Random(seed)

        self._picked_i = {}
        self._picked_p = {}

    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """
        return "corral_rejection"
    
    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.

        See the base class for more information
        """        
        return {"eta": self._eta_init, "f":self._fix_count, "B": [ b.family for b in self._base_learners ] }

    def predict(self, key, context, actions) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        if self._fix_count is None or self._learn_count < self._fix_count:
            picked_i = self._random.choices(range(3), self._p_bars, k=1)[0]
        else:
            picked_i = self._p_bars.index(max(self._p_bars))

        picked_learner = self._base_learners[picked_i]

        self._picked_i[key] = picked_i
        self._picked_p[key] = self._p_bars[picked_i]

        return picked_learner.predict(key, context, actions)

    def learn(self, key, context, action, reward, probability) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """

        self._learn_count += 1

        loss = 1-reward

        assert  0 <= loss and loss <= 1, "The current Corral implementation assumes a loss between 0 and 1"

        picked_i = self._picked_i.pop(key)
        picked_p = self._picked_p.pop(key)

        self._base_learners[picked_i].learn(key, context, action, reward, probability)

        if self._fix_count is None or self._learn_count < self._fix_count:
            self._ps     = list(self._log_barrier_omd([ loss/picked_p * int(i==picked_i) for i in range(len(self._base_learners)) ]))
            self._p_bars = [ (1-self._gamma)*p + self._gamma*1/len(self._base_learners) for p in self._ps ]

            for i in range(len(self._p_bars)):
                if 1/self._p_bars[i] > self._rhos[i]:
                    self._rhos[i] = 2/self._p_bars[i]
                    self._etas[i] *= self._beta

    def _log_barrier_omd(self, losses) -> Sequence[float]:

        f  = lambda l: float(sum( [ 1/((1/p) + eta*(loss-l)) for p, eta, loss in zip(self._ps, self._etas, losses)]))
        df = lambda l: float(sum( [ eta/((1/p) + eta*(loss-l))**2 for p, eta, loss in zip(self._ps, self._etas, losses)]))

        denom_zeros = [ ((-1/p)-(eta*loss))/-eta for p, eta, loss in zip(self._ps, self._etas, losses) ]

        min_loss = min(losses)
        max_loss = max(losses)

        precision = 4

        def newtons_zero(l,r) -> Optional[float]:
            """Use Newton's method to calculate the root."""
            
            #depending on scales this check may fail though that seems unlikely
            if (f(l+.0001)-1) * (f(r-.00001)-1) >= 0:
                return None

            i = 0
            x = (l+r)/2

            while True:
                i += 1

                if df(x) == 0:
                    raise Exception(f'Something went wrong in Corral (0) {self._ps}, {self._etas}, {losses}, {x}')

                x -= (f(x)-1)/df(x)

                if round(f(x),precision) == 1:
                    return x

        lmbda: Optional[float] = None

        if min_loss == max_loss:
            lmbda = min_loss
        elif min_loss not in denom_zeros and round(f(min_loss),precision) == 1:
            lmbda = min_loss
        elif max_loss not in denom_zeros and round(f(max_loss),precision) == 1:
            lmbda = max_loss
        else:
            brackets = list(sorted(filter(lambda z: min_loss <= z and z <= max_loss, set(denom_zeros + [min_loss, max_loss]))))

            for l_brack, r_brack in zip(brackets[:-1], brackets[1:]):
                lmbda = newtons_zero(l_brack, r_brack)
                if lmbda is not None: break

        if lmbda is None:
            raise Exception(f'Something went wrong in Corral (None) {self._ps}, {self._etas}, {losses}')

        return [ max(1/((1/p) + eta*(loss-lmbda)),.00001) for p, eta, loss in zip(self._ps, self._etas, losses)]

class CorralEnsemble:
    """This is modified implementation of the Agarwal et al. (2017) Corral algorithm.

    This algorithm assumes that the reward distribution has support in [0,1].
    This algorithm also implements the remark on pg. 8 in order to improve 
    learning efficiency when multiple bandits select the same action.

    References:
        Agarwal, Alekh, Haipeng Luo, Behnam Neyshabur, and Robert E. Schapire. 
        "Corralling a band of bandit algorithms." In Conference on Learning 
        Theory, pp. 12-38. PMLR, 2017.
    """

    def __init__(self,
        max_memories: int,
        learn_distance: bool,
        epsilon: float,
        eta : float,
        fix_count: int = None,
        T: float = math.inf,
        seed: int = None) -> None:
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

        self._base_learners: Sequence[Learner] = [
            MemorizedLearner(epsilon, max_memories, learn_distance),
            ResidualLearner(epsilon, max_memories, learn_distance),
            VowpalLearner(epsilon=epsilon,seed=seed)
        ]

        M = 3

        self._gamma = 1/T
        self._beta  = 1/math.exp(1/math.log(T))

        self._eta_init = eta
        self._etas     = [ eta ] * M
        self._rhos     = [ float(2*M) ] * M
        self._ps       = [ 1/M ] * M
        self._p_bars   = [ 1/M ] * M

        self._fix_count = fix_count
        self._learn_count = 0

        self._random = random.Random(seed)

        self._picked_i = {}
        self._picked_p = {}

    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """
        return "corral_ensemble"
    
    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.

        See the base class for more information
        """        
        return {"eta": self._eta_init, "f":self._fix_count, "B": [ b.family for b in self._base_learners ] }

    def predict(self, key, context, actions) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        if self._fix_count is None or self._learn_count < self._fix_count:
            picked_i = self._random.choices(range(3), self._p_bars, k=1)[0]
        else:
            picked_i = self._p_bars.index(max(self._p_bars))

        picked_learner = self._base_learners[picked_i]

        self._picked_i[key] = picked_i
        self._picked_p[key] = self._p_bars[picked_i]

        if picked_i != 2:
            #this learner breaks if you don't call predict before learn
            self._base_learners[2].predict(key, context, actions)

        return picked_learner.predict(key, context, actions)

    def learn(self, key, context, action, reward, probability) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """

        self._learn_count += 1

        loss = 1-reward

        assert  0 <= loss and loss <= 1, "The current Corral implementation assumes a loss between 0 and 1"

        picked_i = self._picked_i.pop(key)
        picked_p = self._picked_p.pop(key)

        #pass everything to memorized_learner
        #pass only picked stuff to residual_learner
        #pass everything to vowpal learner

        self._base_learners[0].learn(key, context, action, reward, probability)
        if picked_i == 1: self._base_learners[1].learn(key, context, action, reward, probability)
        self._base_learners[2].learn(key, context, action, reward, probability)

        if self._fix_count is None or self._learn_count < self._fix_count:
            self._ps     = list(self._log_barrier_omd([ loss/picked_p * int(i==picked_i) for i in range(len(self._base_learners)) ]))
            self._p_bars = [ (1-self._gamma)*p + self._gamma*1/len(self._base_learners) for p in self._ps ]

            for i in range(len(self._p_bars)):
                if 1/self._p_bars[i] > self._rhos[i]:
                    self._rhos[i] = 2/self._p_bars[i]
                    self._etas[i] *= self._beta

    def _log_barrier_omd(self, losses) -> Sequence[float]:

        f  = lambda l: float(sum( [ 1/((1/p) + eta*(loss-l)) for p, eta, loss in zip(self._ps, self._etas, losses)]))
        df = lambda l: float(sum( [ eta/((1/p) + eta*(loss-l))**2 for p, eta, loss in zip(self._ps, self._etas, losses)]))

        denom_zeros = [ ((-1/p)-(eta*loss))/-eta for p, eta, loss in zip(self._ps, self._etas, losses) ]

        min_loss = min(losses)
        max_loss = max(losses)

        precision = 4

        def newtons_zero(l,r) -> Optional[float]:
            """Use Newton's method to calculate the root."""
            
            #depending on scales this check may fail though that seems unlikely
            if (f(l+.0001)-1) * (f(r-.00001)-1) >= 0:
                return None

            i = 0
            x = (l+r)/2

            while True:
                i += 1

                if df(x) == 0:
                    raise Exception(f'Something went wrong in Corral (0) {self._ps}, {self._etas}, {losses}, {x}')

                x -= (f(x)-1)/df(x)

                if round(f(x),precision) == 1:
                    return x

        lmbda: Optional[float] = None

        if min_loss == max_loss:
            lmbda = min_loss
        elif min_loss not in denom_zeros and round(f(min_loss),precision) == 1:
            lmbda = min_loss
        elif max_loss not in denom_zeros and round(f(max_loss),precision) == 1:
            lmbda = max_loss
        else:
            brackets = list(sorted(filter(lambda z: min_loss <= z and z <= max_loss, set(denom_zeros + [min_loss, max_loss]))))

            for l_brack, r_brack in zip(brackets[:-1], brackets[1:]):
                lmbda = newtons_zero(l_brack, r_brack)
                if lmbda is not None: break

        if lmbda is None:
            raise Exception(f'Something went wrong in Corral (None) {self._ps}, {self._etas}, {losses}')

        return [ max(1/((1/p) + eta*(loss-lmbda)),.00001) for p, eta, loss in zip(self._ps, self._etas, losses)]

class CorralOffPolicy:
    """This is a modified implementation of the Agarwal et al. (2017) Corral algorithm.

    This algorithm assumes that the reward distribution has support in [0,1] and that all learners can learn off-policy.

    References:
        Agarwal, Alekh, Haipeng Luo, Behnam Neyshabur, and Robert E. Schapire. 
        "Corralling a band of bandit algorithms." In Conference on Learning 
        Theory, pp. 12-38. PMLR, 2017.
    """

    def __init__(self,
        epsilon: float,
        max_memories: int,
        learn_distance: bool,
        eta : float,
        T: float = math.inf,
        d: int = 1, 
        c: int = 10,
        seed: int = 1) -> None:
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

        assert 0 <= epsilon and epsilon <= 1
        assert 1 <= max_memories

        learn_distance = True

        self._base_learners: Sequence[Learner] = [
            VowpalLearner(epsilon=epsilon,seed=seed),
            MemorizedLearner(epsilon, max_memories, learn_distance, d=d, c=c),
#            ResidualLearner(epsilon, max_memories, learn_distance, d=d, c=c)
        ]

        M = len(self._base_learners)

        self._gamma = 1/T
        self._beta  = 1/math.exp(1/math.log(T))

        self._eta_init = eta
        self._etas     = [ eta ] * M
        self._rhos     = [ float(2*M) ] * M
        self._ps       = [ 1/M ] * M
        self._p_bars   = [ 1/M ] * M

        self._learn_count = 0

        self._random = random.Random(seed)

        self._predicts: Dict[int, Sequence[float]] = {}
        self._actions : Dict[int, Sequence[Any]] = {}

        self._full_expected_loss = [0]*M
        self._i = 0

    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """
        return "corral_off_policy"

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.

        See the base class for more information
        """        
        return {"eta": self._eta_init, "B": [ b.family for b in self._base_learners ] }

    def predict(self, key, context, actions) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        self._i += 1

        action_predicts  = [0] * len(actions)
        learner_predicts = [learner.predict(key,context,actions) for learner in self._base_learners]

        self._predicts[key] = learner_predicts
        self._actions [key] = actions

        for learner_p_bar, learner_predict in zip(self._p_bars, learner_predicts):
            for action_index, learner_action_prob in enumerate(learner_predict):
                action_predicts[action_index] += learner_action_prob * learner_p_bar

        assert round(sum(action_predicts),3) == 1

        return action_predicts

    def learn(self, key, context, action, reward, probability) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """

        self._learn_count += 1

        loss = 1-reward

        assert  0 <= loss and loss <= 1, "The current Corral implementation assumes a loss between 0 and 1"

        for base_learner in self._base_learners:
            base_learner.learn(key, context, action, reward, probability)

        actions             = self._actions.pop(key)
        learner_predicts    = self._predicts.pop(key)
        picked_action_index = actions.index(action) 
        base_probabilities  = [ base_predict[picked_action_index] for base_predict in learner_predicts ] 

        expected_loss_estimates = [ loss * base_probability/probability for base_probability in base_probabilities ] 

        for i,l in enumerate(expected_loss_estimates):
            self._full_expected_loss[i] = (1-1/self._i) * self._full_expected_loss[i] + (1/self._i) * l  

        base_predict_data = { f"predict_{i}": learner_predicts[i][picked_action_index] for i in range(len(self._base_learners)) }
        base_pbar_data    = { f"pbar_{i}"   : self._p_bars[i]                          for i in range(len(self._base_learners)) }
        predict_data      = { "predict"     : probability, **base_predict_data, **base_pbar_data }

        self._ps     = self._log_barrier_omd(expected_loss_estimates)
        self._p_bars = [ (1-self._gamma)*p + self._gamma*1/len(self._base_learners) for p in self._ps ]

        # if self._i % 50 == 0:
        #     print(self._p_bars)
        #     print(self._full_expected_loss)
        #     print("")

        for i in range(len(self._p_bars)):
            if 1/self._p_bars[i] > self._rhos[i]:
                self._rhos[i] = 2/self._p_bars[i]
                self._etas[i] *= self._beta

        return { k:round(v,4) for k,v in {**predict_data, **base_predict_data, **base_pbar_data}.items() }

    def _log_barrier_omd(self, losses) -> List[float]:

        f  = lambda l: float(sum( [ 1/((1/p) + eta*(loss-l)) for p, eta, loss in zip(self._ps, self._etas, losses)]))
        df = lambda l: float(sum( [ eta/((1/p) + eta*(loss-l))**2 for p, eta, loss in zip(self._ps, self._etas, losses)]))

        denom_zeros = [ ((-1/p)-(eta*loss))/-eta for p, eta, loss in zip(self._ps, self._etas, losses) ]

        min_loss = min(losses)
        max_loss = max(losses)

        precision = 4

        def newtons_zero(l,r) -> Optional[float]:
            """Use Newton's method to calculate the root."""

            #depending on scales this check may fail though that seems unlikely
            if (f(l+.0001)-1) * (f(r-.00001)-1) >= 0:
                return None

            i = 0
            x = (l+r)/2

            while True:
                i += 1

                if df(x) == 0:
                    raise Exception(f'Something went wrong in Corral (0) {self._ps}, {self._etas}, {losses}, {x}')

                x -= (f(x)-1)/df(x)

                if round(f(x),precision) == 1:
                    return x

        lmbda: Optional[float] = None

        if min_loss == max_loss:
            lmbda = min_loss
        elif min_loss not in denom_zeros and round(f(min_loss),precision) == 1:
            lmbda = min_loss
        elif max_loss not in denom_zeros and round(f(max_loss),precision) == 1:
            lmbda = max_loss
        else:
            brackets = list(sorted(filter(lambda z: min_loss <= z and z <= max_loss, set(denom_zeros + [min_loss, max_loss]))))

            for l_brack, r_brack in zip(brackets[:-1], brackets[1:]):
                lmbda = newtons_zero(l_brack, r_brack)
                if lmbda is not None: break

        if lmbda is None:
            raise Exception(f'Something went wrong in Corral (None) {self._ps}, {self._etas}, {losses}')

        return [ max(1/((1/p) + eta*(loss-lmbda)),.00001) for p, eta, loss in zip(self._ps, self._etas, losses)]