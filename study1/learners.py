import time
import random
import math

from itertools import count
from typing import Hashable, Sequence, Dict, Any, Optional

from memory import CMT

logn = None
bits = 20

class CMT_Implemented:
    class LogisticModel:
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
                x      = enumerate(x)
                xprime = enumerate(xprime)

            if not isinstance(a[0],tuple):
                a      = enumerate(a)
                aprime = enumerate(aprime)

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

                if not isinstance(a[0],tuple):
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

    def __init__(self, max_memories: int = 1000, learn_dist: bool = True, signal_type:str = 'se') -> None:

        self._learn_dist   = learn_dist
        self._max_memories = max_memories
        self._signal_type  = signal_type

    @property
    def params(self):
        return {'m': self._max_memories, 'b': bits, 'ld': self._learn_dist, 'sig': self._signal_type }

    def init(self):
        router_factory = CMT_Implemented.LogisticModel 
        scorer         = CMT_Implemented.LearnedEuclideanDistance(self._learn_dist)
        random_state   = random.Random(1337)
        ords           = random.Random(2112)

        self.mem = CMT(router_factory, scorer, alpha=0.25, c=40, d=1, randomState=random_state, optimizedDeleteRandomState=ords, maxMemories=self._max_memories)

    def query(self, context, actions, default = None):
        for action in actions:
            (_, z) = self.mem.query(self._flat(context,action), 1, 0)
            if len(z) >  0                        : yield (action, z[0][1])
            if len(z) == 0 and default is not None: yield (action, default)

    def update(self, context, action, value):

        x = self._flat(context, action)

        (u,z) = self.mem.query(x, k=2, epsilon=1)

        if len(z) > 0:
            megalr = 0.1
            newval = (1.0 - megalr) * z[0][1] + megalr * value

            self.mem.updateomega(z[0][0], newval)
            self.mem.update(u, x, z, self._signal(value, newval))

        if x in self.mem.leafbykey:
            self.mem.delete(x)

        self.mem.insert(x, value)

    def _flat(self, context, action):
        if isinstance(context,dict):
            return (tuple(context.items()), action)
        else:
            return (context, action)

    def _signal(self, obs, prd):

        if self._signal_type == 'se':
            return 1-(prd-obs)**2

        if self._signal_type == 'l1':
            deltarvw    = sorted([-1, obs                           , 1])[1]
            deltarcombo = sorted([-1, obs-prd                       , 1])[1]
            rupdate     = sorted([0 , abs(deltarvw)-abs(deltarcombo)   ])[1]
            return rupdate

        if self._signal_type == "pct":
            if obs == 0:
                return float(prd==0)
            else:
                return sorted([0, 1. - abs(obs-prd)/abs(prd)])[1]

        raise Exception(f"Unrecognized signal type: {self._signal}")

class MemorizedLearner:
    
    def __init__(self, epsilon: float, max_memories: int = 1000, learn_dist: bool = True, signal:str = 'se') -> None:

        self._epsilon = epsilon
        self._i       = 0

        self.mem = CMT_Implemented(max_memories, learn_dist, signal)

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

        for action, value in self.mem.query(context, actions):
            if value > greedy_r: (greedy_a, greedy_r) = (action, value)

        ga   = actions.index(greedy_a)
        minp = self._epsilon / len(actions)

        if logn and self._i % logn == 0:
           print(f"{self._i}. prediction time {round(time.time()-predict_start, 2)}")

        return [ minp if i != ga else minp+(1-self._epsilon) for i in range(len(actions)) ]

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""
        self.mem.update(context, action, reward)

class ResidualLearner:
    def __init__(self, epsilon: float, max_memories: int, learn_dist: bool, signal:str = 'l1'):

        self._epsilon = epsilon
        self.mem = CMT_Implemented(max_memories, learn_dist, signal)

        self._i        = 0
        self._predicts = {}

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
        assert isinstance(context, (tuple, dict))

        if isinstance(context, tuple):
            context_dict = dict(enumerate(context))
        else:
            context_dict = context

        return '\n'.join([
            'shared |s ' + ' '.join([ f'{k+1}:{v}' for k, v in context_dict.items() ]),
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
        deltas = []

        for _,value in self.mem.query(context,actions,0):
            deltas.append(value)            

        ga   = min(((p + dp, n) for p, dp, n in zip(predicts, deltas, range(len(actions)))))[1]
        minp = self._epsilon / len(actions)

        self._predicts[key] = (predicts, actions)

        if logn and self._i % logn == 0:
            print(f"{self._i}. prediction time {round(time.time()-predict_start, 2)}")

        return [ minp if i != ga else minp+(1-self._epsilon) for i in range(len(actions)) ]

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""

        (predicts, actions) = self._predicts.pop(key)

        act_ind = actions.index(action)
        prd_loss = predicts[act_ind]

        obs_loss  = -reward
        obs_resid = obs_loss-prd_loss

        self.vw.learn(self.toadf(context, actions, (act_ind, obs_loss, probability)))
        self.mem.update(context, action, obs_resid)

class CorralRejectionLearner:
    """This is an implementation of the Agarwal et al. (2017) Corral algorithm.

    This algorithm assumes that the reward distribution has support in [0,1].
    This algorithm also implements the remark on pg. 8 in order to improve 
    learning efficiency when multiple bandits select the same action.

    References:
        Agarwal, Alekh, Haipeng Luo, Behnam Neyshabur, and Robert E. Schapire. 
        "Corralling a band of bandit algorithms." In Conference on Learning 
        Theory, pp. 12-38. PMLR, 2017.
    """

    def __init__(self, base_learners,
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

        self._base_learners = base_learners

        M = len(self._base_learners)

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
            try:
                learner.init()
            except:
                pass

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
            picked_i       = self._random.choices(range(len(self._base_learners)), self._p_bars, k=1)[0]
        else:
            picked_i       = self._p_bars.index(max(self._p_bars))
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

            for i in range(len(self._base_learners)):
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