import time
import random
import math

from itertools import count
from typing import Hashable, Sequence, Dict, Any

from memory import CMT

logn = None
bits = 20

class CMT_Implementation_1:
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
        router_factory = CMT_Implementation_1.LogisticModel 
        scorer         = CMT_Implementation_1.LearnedEuclideanDistance(self._learn_dist)
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
    
    def __init__(self, epsilon: float, max_memories: int = 1000, learn_dist: bool = True) -> None:

        self._epsilon = epsilon
        self._i       = 0

        self.mem = CMT_Implementation_1(max_memories, learn_dist)

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
        self.mem = CMT_Implementation_1(max_memories, learn_dist, signal)

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