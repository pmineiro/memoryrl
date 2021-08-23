import time
import random
import math

from itertools import count
from typing import Hashable, Sequence, Dict, Any, Optional, Tuple, Union, List

from vowpalwabbit import pyvw 
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction import FeatureHasher

from coba.encodings import InteractionTermsEncoder
from coba.learners import VowpalLearner, Learner, CorralLearner

from memory import CMT

logn = 500
bits = 20

class CMT_Implemented:

    class MemoryKey:

        def __init__(self, context, action) -> None:

            features = self._featurize(context, action)

            if isinstance(features[0],tuple):
                self._features = FeatureHasher(n_features=2**18, input_type='pair').fit_transform([features])
            else:
                #doing this because some code later that assumes we're working with sparse matrices
                self._features = sp.csr_matrix((features,([0]*len(features),range(len(features)))), shape=(1,len(features)))

            self._hash = hash((context,action))

        def features(self):
            return self._features

        def _featurize(self, context, action):
            return InteractionTermsEncoder(['x','a','xa','xxa']).encode(x=context,a=action)            

        def __hash__(self) -> int:
            return self._hash

    class LogisticModel_VW:
        def __init__(self, *args, **kwargs):
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

        def __reduce__(self):
            return (CMT_Implemented.LogisticModel_VW,())

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
            if self.vw and dp.nnz > 0:
                example    = pyvw.example(self.vw, {"x": list(zip(map(str,dp.indices), dp.data**2))})
                prediction = self.vw.predict(example)
            else:
                prediction = 0
            ee[4] = time.time()

            for i,s,e in zip(count(),ss,ee):
                self.tt[i] += e-s

            initial = -0.01 * dp_dot_dp
            return initial + prediction

        def update(self, xraw, z, r):
            if not self.vw: return

            if r > 0 and len(z) > 1:
                X   = xraw.features()
                Xp  = z[0][0].features()
                Xpp = z[1][0].features()

                dp  = self.diff(X, Xp)
                dpp = self.diff(X, Xpp)

                mat = dp.power(2) - dpp.power(2)
                data = mat.data
                keys = map(str,mat.indices)

                start = time.time()
                initial  = 0.01 * (self.inner(dp,dp) - self.inner(dpp,dpp))
                example = pyvw.example(self.vw, {"x": list(zip(keys,data))})
                example.set_label_string(f"1 {r} {initial}")
                self.vw.learn(example)
                self.tt[4] += time.time()-start

        def __reduce__(self):
            return (CMT_Implemented.LearnedEuclideanDistance, (self.learn,) )

    class MarksRewardLearner:

        def __init__(self,learn=True):

            from sklearn.linear_model import SGDRegressor

            self.clf    = SGDRegressor(loss="squared_loss", learning_rate='invscaling', eta0=1, power_t=1, average=True)
            self.is_fit = False
            self.learn  = learn

        def model_features(self, vec1, vec2):
            return sp.hstack([vec1,vec1.multiply(vec2)/(vec1.power(2).sum() * vec2.power(2).sum())**1/2]) 

        def normed_linear_prod(self, vec1, vec2):
            return vec1.multiply(vec2).sum() / (vec1.power(2).sum() * vec2.power(2).sum())**1/2

        def predict(self, trigger, memory):

            ec1 = trigger.features()
            ec2 = memory[0].features()

            feat = self.model_features(ec1,ec2)
            init = self.normed_linear_prod(ec1,ec2)
            pred = 0 if not self.is_fit else self.clf.predict(feat)[0]

            return init + pred

        def update(self, trigger, memory, reward):
            if not self.learn: return

            ec1 = trigger.features()
            ec2 = memory[0][0].features()

            feat = self.model_features(ec1,ec2)
            init = self.normed_linear_prod(ec1,ec2)

            self.is_fit = True
            self.clf.partial_fit(feat, [reward-init])

    def __init__(self, max_memories: int = 1000, learn_dist: bool = True, signal_type:str = 'se', router_type:str = 'sk', scorer_type:str = 'vw', c=10, d=1, megalr=0.1) -> None:

        self._max_memories = max_memories
        self._learn_dist   = learn_dist
        self._signal_type  = signal_type
        self._router_type  = router_type
        self._scorer_type  = scorer_type
        self._c            = c
        self._d            = d
        self._megalr       = megalr

        router_factory = CMT_Implemented.LogisticModel_SK if self._router_type == 'sk' else CMT_Implemented.LogisticModel_VW
        scorer         = CMT_Implemented.LearnedEuclideanDistance(self._learn_dist) if self._scorer_type == 'vw' else CMT_Implemented.MarksRewardLearner(self._learn_dist)

        random_state   = random.Random(1337)
        ords           = random.Random(2112)

        self.mem = CMT(router_factory, scorer, alpha=0.25, c=self._c, d=self._d, randomState=random_state, optimizedDeleteRandomState=ords, maxMemories=self._max_memories)

    @property
    def params(self):
        return { 'm': self._max_memories, 'b': bits, 'ld': self._learn_dist, 'sig': self._signal_type, 'rt': self._router_type, 'st': self._scorer_type, 'd': self._d, 'c': self._c, 'mlr': self._megalr }

    def query(self, context: Hashable, actions: Sequence[Hashable], default = None, topk:int=1):

        memories = []

        for action in actions:
            trigger = self._mem_key(context, action)
            (_, z) = self.mem.query(trigger, topk, 0)
            memories.extend([ zz[1] for zz in z ] or [default])

        return memories

    def update(self, context: Hashable, action: Hashable, observation: Union[float, np.ndarray]):

        trigger = self._mem_key(context, action)
        (u,z) = self.mem.query(trigger, k=2, epsilon=1) #we select randomly with probability 1-epsilon

        if len(z) > 0:

            memory = z[0][1] 

            if self._megalr > 0:
                memory += self._megalr * ( observation-memory )
                self.mem.updateomega(z[0][0], memory)

            self.mem.update(u, trigger, z, self._error_signal(observation, memory))

        if trigger in self.mem.leafbykey:
            self.mem.delete(trigger)

        self.mem.insert(trigger, observation)

    def _error_signal(self, obs, prd):

        if self._signal_type == 'se':
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

    def __init__(self, epsilon: float, max_memories: int = 1000, learn_dist: bool = True, c=10, d=1, signal:str = 'se', router: str = 'sk', scorer: str='vw', megalr=0.1) -> None:

        assert 0 <= epsilon and epsilon <= 1
        assert 1 <= max_memories

        self._epsilon = epsilon
        self._i       = 0

        self.mem = CMT_Implemented(max_memories, learn_dist, signal, router, scorer, c=c, d=d, megalr=megalr)

        self._times = [0, 0]

    @property
    def family(self) -> str:
        return "CMT_Memorized"

    @property
    def params(self) -> Dict[str,Any]:
        return { 'e':self._epsilon,  **self.mem.params }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        self._i += 1

        if logn and self._i % logn == 0:
           #print(self.mem.query(context, actions))
           #print(self.mem.mem.f.__class__.__name__)
           print(f"MEM {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"MEM {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        predict_start = time.time()
        
        (greedy_a, greedy_r) = actions[0], -math.inf

        for action, remembered_value in zip(actions, self.mem.query(context, actions)):
            if remembered_value is None: continue
            if remembered_value > greedy_r: (greedy_a, greedy_r) = (action, remembered_value)

        ga   = actions.index(greedy_a)
        minp = self._epsilon / len(actions)

        self._times[0] += time.time()-predict_start

        return [ minp if i != ga else minp+(1-self._epsilon) for i in range(len(actions)) ], None

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, info: Any) -> None:
        """Learn about the result of an action that was taken in a context."""
        learn_start = time.time()
        self.mem.update(context, action, reward)
        self._times[1] += time.time()-learn_start

class ResidualLearner:
    def __init__(self, epsilon: float, max_memories: int, learn_dist: bool = True, c=10, d=1, signal:str = 're', router:str ='sk', scorer:str = 'vw', megalr:float = 0.1):

        assert 0 <= epsilon and epsilon <= 1
        assert 1 <= max_memories

        from vowpalwabbit import pyvw

        self._args = (epsilon, max_memories, learn_dist, c, d, signal, router, scorer, megalr)

        self._epsilon = epsilon
        self._max_memories = max_memories
        self._learn_dist = learn_dist
        self._c = c
        self._d = d
        self._signal = signal
        self._router = router
        
        self.mem = CMT_Implemented(max_memories, learn_dist, signal, router, scorer, c=c, d=d, megalr=megalr)
        self.vw  = pyvw.vw(f'--quiet -b {bits} --cb_adf -q sa --cubic ssa --ignore_linear s')

        self._i        = 0
        self._predicts = {}
        self._times    = [0,0]

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

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Tuple[Sequence[float], Any]:
        """Choose which action index to take."""

        predict_start = time.time()
        self._i += 1

        predicted_losses = self.vw.predict(self.toadf(context, actions))
        rememberd_deltas = self.mem.query(context, actions, 0)

        ga   = min(((pl+rd, n) for pl, rd, n in zip(predicted_losses, rememberd_deltas, range(len(actions)))))[1]
        minp = self._epsilon / len(actions)

        self._times[0] += time.time()-predict_start

        if logn and self._i % logn == 0:
           print(f"RES {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"RES {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        prediction = [ minp if i != ga else minp+(1-self._epsilon) for i in range(len(actions)) ]
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

        self.vw.learn(self.toadf(context, actions, (act_ind, obs_loss, probability)))
        self.mem.update(context, action, obs_resid)

        self._times[1] += time.time()-learn_start
    
    def __reduce__(self):
        return (ResidualLearner,self._args)

class CorralOffPolicy_Old:
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

        self._base_learners: Sequence[Learner] = [
            VowpalLearner(epsilon=epsilon,seed=seed),
            MemorizedLearner(epsilon, max_memories, learn_distance, d=d, c=c),
            #ResidualLearner(epsilon, max_memories, learn_distance, d=d, c=c)
        ]

        M = len(self._base_learners)

        self._gamma = 1/T
        self._beta  = 1/math.exp(1/math.log(T))

        self._eta_init = eta
        self._etas     = [ eta ] * M
        self._rhos     = [ float(2*M) ] * M
        self._ps       = [ 1/M ] * M
        self._p_bars   = [ 1/M ] * M

        self._random = random.Random(seed)

        self._predicts: Dict[int, Sequence[float]] = {}
        self._actions : Dict[int, Sequence[Any]] = {}

        self._full_expected_rwd = [0]*M
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

    def predict(self, context, actions) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        self._i += 1

        action_predicts  = [0] * len(actions)
        learner_predicts, learner_info = zip(*[learner.predict(context,actions) for learner in self._base_learners])

        self._predicts[self._i] = learner_predicts
        self._actions [self._i] = actions

        for learner_p_bar, learner_predict in zip(self._p_bars, learner_predicts):
            for action_index, learner_action_prob in enumerate(learner_predict):
                action_predicts[action_index] += learner_action_prob * learner_p_bar

        assert round(sum(action_predicts),3) == 1

        return action_predicts, learner_info

    def learn(self, context, action, reward, probability, info) -> None:
        """Learn from the given interaction.

        Args:
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """

        loss = 1-reward

        assert 0 <= loss and loss <= 1, "The current Corral implementation assumes a loss between 0 and 1"

        for base_learner, base_info in zip(self._base_learners, info):
            base_learner.learn(context, action, reward, probability, base_info)

        actions             = self._actions.pop(self._i)
        learner_predicts    = self._predicts.pop(self._i)
        picked_action_index = actions.index(action) 
        base_probabilities  = [ base_predict[picked_action_index] for base_predict in learner_predicts ] 

        expected_lss_estimates = [ loss * base_probability/probability for base_probability in base_probabilities ]
        expected_rwd_estimates = [ reward * base_probability/probability for base_probability in base_probabilities ]

        for i,l in enumerate(expected_rwd_estimates):
            self._full_expected_rwd[i] = (1-1/self._i) * self._full_expected_rwd[i] + (1/self._i) * l  

        base_predict_data = { f"predict_{i}": learner_predicts[i][picked_action_index] for i in range(len(self._base_learners)) }
        base_pbar_data    = { f"pbar_{i}"   : self._p_bars[i]                          for i in range(len(self._base_learners)) }
        predict_data      = { "predict"     : probability, **base_predict_data, **base_pbar_data }

        self._ps     = self._log_barrier_omd(expected_lss_estimates)
        self._p_bars = [ (1-self._gamma)*p + self._gamma*1/len(self._base_learners) for p in self._ps ]

        if self._i % int(logn/5) == 0:
            print(self._p_bars)
            print(self._full_expected_rwd)
            print("")

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

class LoggedCorralLearner(CorralLearner):
    """This is a modified implementation of the Agarwal et al. (2017) Corral algorithm.

    This algorithm assumes that the reward distribution has support in [0,1] and that all learners can learn off-policy.

    References:
        Agarwal, Alekh, Haipeng Luo, Behnam Neyshabur, and Robert E. Schapire. 
        "Corralling a band of bandit algorithms." In Conference on Learning 
        Theory, pp. 12-38. PMLR, 2017.
    """

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

        if self._i % int(logn/5) == 0:
            print(self._p_bars)
            print(self._full_expected_rwd)
            print("")

        return super().learn(context, action, reward, probability, info)

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
