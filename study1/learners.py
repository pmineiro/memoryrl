import time
import random
import math

from itertools import count
from typing import Hashable, Sequence, Dict, Any, Optional, Tuple, Union

import numpy as np

from coba.learners import VowpalLearner

from memory import CMT

logn = None
bits = 20

class CMT_Implemented:
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

            from sklearn.feature_extraction import FeatureHasher
            from sklearn.linear_model import SGDClassifier
            
            self.clf  = SGDClassifier(loss="log", average=True)
            self.hash = FeatureHasher(input_type="pair")
            self.is_fit = False

        def predict(self, xraw): 
            if self.is_fit:
                return self.clf.predict(self._domain(*xraw))[0]
            else:
                return 1

        def update(self, xraw, y, w):
            self.is_fit = True
            self.clf.partial_fit(self._domain(*xraw), [y], sample_weight=[w], classes=[-1,1])
        
        def _flat_outer(self, vec1: Sequence[Tuple[str,float]], vec2: Sequence[Tuple[str,float]]) -> Sequence[Tuple[str,float]]:
            return [ (key1+'_'+key2, val1*val2) for key1,val1 in vec1 for key2,val2 in vec2 ]

        def _domain(self, x, a):

            if isinstance(x[0], tuple):
                x   = [ (f"x{i}", v) for i,v in zip(*x) ]
                a   = [ (f"a{i}", v) for i,v in enumerate(a) if v != 0 ]
                xa  = self._flat_outer(x,a)
                xxa = self._flat_outer(self._flat_outer(x,x),a)

                features = self.hash.transform([x+a+xa+xxa])
            else:
                x = list(x)
                a = list(a)
                xa = np.outer(x,a).flatten().tolist()
                xxa = np.outer(np.outer(x,x),a).flatten().tolist()
                features = [x+a+xa+xxa]

            return features

    class LearnedEuclideanDistance:

        def __init__(self,learn=True):

            from vowpalwabbit import pyvw

            self.vw    = None if not learn else pyvw.vw(f'--quiet -b {bits} --noconstant --loss_function logistic --link=glf1')            
            self.tt    = [0]*6
            self.learn = learn

        def outer(self, vec1: Dict[str,float], vec2: Dict[str,float]) -> Dict[str,float]:
            return { key1+'_'+key2: val1*val2 for key1,val1 in vec1.items() for key2,val2 in vec2.items() }

        def diff(self, vec1: Dict[str,float], vec2: Dict[str,float]) -> Dict[str,float]:
            
            diff = {}
            
            for key in set(vec1.keys()) | set(vec2.keys()):
                if key in vec1 and key in vec2:
                    diff[key] = vec1[key]-vec2[key]
                elif key in vec1:
                    diff[key] = vec1[key]
                else:
                    diff[key] = -vec2[key]

            return diff

        def inner(self, vec1: Dict[str,float], vec2: Dict[str,float]) -> float:

            keys = set(vec1.keys()) & set(vec2.keys())

            return sum([ vec1[key]*vec2[key] for key in keys ])

        def predict(self, xraw, z):

            ss = [0]*6
            ee = [0]*6

            xprimeraw = z[0]

            (x     , a     ) = xraw
            (xprime, aprime) = xprimeraw

            ss[0] = time.time()
            
            if not isinstance(x[0],tuple):
                x      = list(enumerate(x))
                xprime = list(enumerate(xprime))
            else:
                x      = list(zip(*x))
                xprime = list(zip(*xprime))

            if len(a) > 0 and not isinstance(a[0],tuple):
                a      = list(enumerate(a))
                aprime = list(enumerate(aprime))

            x      = { "x"+str(key):value for key,value in x if value != 0}
            a      = { "a"+str(key):value for key,value in a if value != 0 }

            xprime = { "x"+str(key):value for key,value in xprime if value != 0}
            aprime = { "a"+str(key):value for key,value in aprime if value != 0 }
            ee[0] = time.time()

            ss[1] = time.time()
            xa      = {**x, **a, **self.outer(x,a) }
            xaprime = {**xprime, **aprime, **self.outer(xprime,aprime) }
            ee[1] = time.time()

            ss[2] = time.time()
            dxa = self.diff(xa,xaprime)
            ee[2] = time.time()

            ss[3] = time.time()
            dxa_dot_dxa = self.inner(dxa, dxa)                            
            ee[3] = time.time()

            ss[4] = time.time()
            if self.vw:
                v = -0.01 * dxa_dot_dxa + self.vw.predict(f' |x ' + ' '.join([f'{k}:{round(v*v,6)}' for k,v in dxa.items()]))
            else:
                v = -0.01 * dxa_dot_dxa
            ee[4] = time.time()

            for i,s,e in zip(count(),ss,ee):
                self.tt[i] += e-s

            return v

        def update(self, xraw, z, r):
            if not self.vw: return

            if r > 0 and len(z) > 1:
                (x     , a     ) = xraw
                (xprime, aprime) = z[0][0]
                (xpp   , app   ) = z[1][0]
                
                if not isinstance(x[0],tuple):
                    x      = enumerate(x)
                    xprime = enumerate(xprime)
                    xpp    = enumerate(xpp)
                else:
                    x      = list(zip(*x))
                    xprime = list(zip(*xprime))
                    xpp    = list(zip(*xpp))

                if len(a) > 0 and not isinstance(a[0],tuple):
                    a      = enumerate(a)
                    aprime = enumerate(aprime)
                    app    = enumerate(app)

                x      = { "x"+str(key):value for key,value in x if value != 0 }
                a      = { "a"+str(key):value for key,value in a if value != 0 }

                xprime = { "x"+str(key):value for key,value in xprime if value != 0 }
                aprime = { "a"+str(key):value for key,value in aprime if value != 0 }

                xpp = { "x"+str(key):value for key,value in xpp if value != 0 }
                app = { "a"+str(key):value for key,value in app if value != 0 }

                xa      = {**x, **a, **self.outer(x,a) }
                xaprime = {**xprime, **aprime, **self.outer(xprime,aprime) }
                xapp    = {**xpp, **app, **self.outer(xpp,app) }

                dxa  = self.diff(xa, xaprime)
                dxap = self.diff(xa, xapp)

                initial = 0.01 * (self.inner(dxa,dxa) - self.inner(dxap,dxap))

                keys = set(dxa.keys()) & set(dxap.keys())

                ex = f'1 {r} {initial} |x ' + ' '.join([f'{key}:{round(dxa[key]**2-dxap[key]**2,6)}' for key in keys])
                self.vw.learn(ex)

    def __init__(self, max_memories: int = 1000, learn_dist: bool = True, signal_type:str = 'se', router_type:str = 'sk', d=1) -> None:

        self._learn_dist   = learn_dist
        self._max_memories = max_memories
        self._signal_type  = signal_type
        self._router_type  = router_type
        self._d = d

    @property
    def params(self):
        return { 'm': self._max_memories, 'b': bits, 'ld': self._learn_dist, 'sig': self._signal_type, 'rt': self._router_type, 'd': self._d }

    def init(self):
        router_factory = CMT_Implemented.LogisticModel_SK if self._router_type == 'sk' else CMT_Implemented.LogisticModel_VW
        
        scorer         = CMT_Implemented.LearnedEuclideanDistance(self._learn_dist)
        random_state   = random.Random(1337)
        ords           = random.Random(2112)

        self.mem = CMT(router_factory, scorer, alpha=0.25, c=10, d=self._d, randomState=random_state, optimizedDeleteRandomState=ords, maxMemories=self._max_memories)

    def query(self, context: Hashable, actions: Sequence[Hashable], default = None):
        
        results = []

        for action in actions:
            (_, z) = self.mem.query((context,action), 1, 0)
            results.append(z[0][1] if len(z) > 0 else default)

        return results

    def update(self, context: Hashable, action: Hashable, observation: Union[float, np.ndarray]):

        memory_trigger = (context, action)

        (u,z) = self.mem.query(memory_trigger, k=2, epsilon=1)

        if len(z) > 0:
            memory = z[0][1] + 0.1 * ( observation-z[0][1] )

            self.mem.updateomega(z[0][0], memory)
            self.mem.update(u, memory_trigger, z, self._error_signal(observation, memory))

        if memory_trigger in self.mem.leafbykey:
            self.mem.delete(memory_trigger)

        self.mem.insert(memory_trigger, observation)

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

class FullFeedbackLearner:
    def __init__(self, epsilon: float, max_memories: int = 1000, learn_dist: bool = True, d=1, signal:str = 'se', router: str = 'sk'):
        self._epsilon = epsilon
        self._i       = 0

        self.mem = CMT_Implemented(max_memories, learn_dist, signal, router, d=d)

        self._full_reward = None
        self._fixed_actions = None

        self._times = [0, 0]

    def init(self):
        self.mem.init()

    @property
    def family(self) -> str:
        return "CMT_FullFeedback"

    @property
    def params(self) -> Dict[str,Any]:
        return { 'e':self._epsilon,  **self.mem.params }

    def predict(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        predict_start = time.time()

        self._fixed_actions = self._fixed_actions or actions
        self._i += 1

        default_action = np.array([1] + [0]*(len(self._fixed_actions)-1))
        action_one_hot = self.mem.query(context, [()], default=default_action)[0]

        #print(action_one_hot)

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
        #ga = actions.index(self._full_reward._labels[key])
        minp = self._epsilon / len(self._fixed_actions)

        self._times[0] += time.time()-predict_start

        if logn and self._i % logn == 0:
           print(f"{self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"{self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        return [ minp if i != ga else minp+(1-self._epsilon) for i in range(len(self._fixed_actions)) ]

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""        

        learn_start = time.time()

        try:
            true_label = self._full_reward._labels[key]
            true_one_hot = np.zeros(len(self._fixed_actions))
            true_one_hot[self._fixed_actions.index(true_label)] = 1

            assert reward == int(action == true_label)
        
        except:
            true_one_hot = np.array([ self._full_reward._rewards[(key,action)] for action in self._fixed_actions ])

        self.mem.update(context, (), true_one_hot)

        self._times[1] += time.time()-learn_start

class MemorizedLearner:
    
    def __init__(self, epsilon: float, max_memories: int = 1000, learn_dist: bool = True, d=1, signal:str = 'se', router: str = 'sk') -> None:

        self._epsilon = epsilon
        self._i       = 0

        self.mem = CMT_Implemented(max_memories, learn_dist, signal, router, d=d)

        self._times = [0, 0]

    def init(self):
        self.mem.init()

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
           print(f"{self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"{self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        return [ minp if i != ga else minp+(1-self._epsilon) for i in range(len(actions)) ]

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""
        learn_start = time.time()
        
        self.mem.update(context, action, reward)
        
        self._times[1] += time.time()-learn_start

class ResidualLearner:
    def __init__(self, epsilon: float, max_memories: int, learn_dist: bool, d=1, signal:str = 're', router:str ='sk'):

        self._epsilon = epsilon
        self.mem      = CMT_Implemented(max_memories, learn_dist, signal, router, d=d)

        self._i        = 0
        self._predicts = {}
        self._times    = [0,0]

    def init(self):
        from vowpalwabbit import pyvw

        self.mem.init()
        self.vw = pyvw.vw(f'--quiet -b {bits} --cb_adf -q sa --cubic ssa --ignore_linear s')

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
           print(f"{self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"{self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

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

    def init(self) -> None:
        for learner in self._base_learners:
            learner.init()

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

                if (i % 30000) == 0:
                    print(i)

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
        return "corral_ensemble"
    
    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.

        See the base class for more information
        """        
        return {"eta": self._eta_init, "f":self._fix_count, "B": [ b.family for b in self._base_learners ] }

    def init(self) -> None:
        for learner in self._base_learners:
            learner.init()

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
        self._base_learners[2].learn(key, context, action, reward, probability*picked_p)

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

                if (i % 30000) == 0:
                    print(i)

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