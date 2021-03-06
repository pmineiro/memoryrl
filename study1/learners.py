import time
import random
import math

from collections import defaultdict
from typing import Hashable, Sequence, Dict, Any

import numpy as np
import torch

from sklearn.exceptions import NotFittedError
from sklearn import linear_model
from torch import nn
from torch import optim

from coba.tools import CobaConfig

from memory import CMT

logn = None
bits = 13

class MemorizedLearner_1:
    class LogisticModel:
        def __init__(self, *args, **kwargs):
            self.vw = None

        def incorporate(self):
            if self.vw is None:
                from os import devnull
                from coba import tools

                #with open(devnull, 'w') as f, tools.redirect_stderr(f):
                from vowpalwabbit import pyvw
                self.vw = pyvw.vw(f'--quiet -b {bits} --loss_function logistic --link=glf1 -q ax --cubic axx')

        def predict(self, xraw):
            self.incorporate()

            (x, a) = xraw
            ex = ' |x ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(x) if v != 0]
            )  + ' |a ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(a) if v != 0]
            )

            return self.vw.predict(ex)

        def update(self, xraw, y, w):
            self.incorporate()

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
        def __init__(self, *args, **kwargs):
            self.vw = None

        def incorporate(self):
            if self.vw is None:
                from vowpalwabbit import pyvw

                self.vw = pyvw.vw(f'--quiet -b {bits} --noconstant --loss_function logistic -qxx --link=glf1')

        def predict(self, xraw, z):
            self.incorporate()

            import numpy as np

            (x, a) = xraw

            (xprimeraw, omegaprime) = z
            (xprime, aprime) = xprimeraw

            xa = np.hstack(( x, a, np.reshape(np.outer(x, a), -1) ))
            xaprime = np.hstack(( xprime, aprime, np.reshape(np.outer(xprime, aprime), -1) ))

            dxa = xa - xaprime
            initial = -0.01 * dxa.dot(dxa)

            ex = f' |x ' + ' '.join([f'{n+1}:{v*v}' for n, v in enumerate(dxa)])
            return initial + self.vw.predict(ex)

        def update(self, xraw, z, r):
            self.incorporate()

            import numpy as np

            if r > 0 and len(z) > 1:
                (x, a) = xraw
                xa = np.hstack(( x, a, np.reshape(np.outer(x, a), -1) ))

                (xprime, aprime) = z[0][0]
                xaprime = np.hstack(( xprime, aprime, np.reshape(np.outer(xprime, aprime), -1) ))
                dxa = xa - xaprime

                (xpp, app) = z[1][0]
                xapp = np.hstack(( xpp, app, np.reshape(np.outer(xpp, app), -1) ))
                dxap = xa - xapp

                initial = 0.01 * (dxa.dot(dxa) - dxap.dot(dxap))

                ex = f'1 {r} {initial} |x ' + ' '.join([f'{n+1}:{v*v-vp*vp}' for n, (v, vp) in enumerate(zip(dxa, dxap))])

    class MarkLearnedEuclideanDistance:
        def __init__(self, *args, **kwargs):
            self.vw = None

        def incorporate(self):
            if self.vw is None:
                from vowpalwabbit import pyvw

                self.vw = pyvw.vw(f'--quiet -b {bits} --noconstant --loss_function logistic --interactions xxa --link=glf1')

        def predict(self, xraw, z):
            self.incorporate()

            import numpy as np

            xprimeraw = z[0]

            (x     , a     ) = xraw
            (xprime, aprime) = xprimeraw

            if isinstance(x[0],tuple):
                x      = defaultdict(int,x)
                xprime = defaultdict(int,xprime)
            else:
                x      = defaultdict(int,enumerate(x))
                xprime = defaultdict(int,enumerate(xprime))

            keys = set(list(x.keys()) + list(xprime.keys()))

            dx = { key:x[key]-xprime[key] for key in keys }
            da = np.hstack(a) - np.hstack(aprime)

            #we've gotten rid of the interaction terms. In practice though these only matter if a==aprime
            if a == aprime:
                dxa_dot_dxa = sum([2*dx[key]*dx[key] for key in keys])
            else:
                dxa_dot_dxa = sum([dx[key]*dx[key] for key in keys]) + sum([x[key]*x[key] for key in keys]) + sum([xprime[key]*xprime[key] for key in keys])
        
            initial = -0.01 * dxa_dot_dxa

            ex = ' |x ' + ' '.join([f'{n+1}:{v*v}' for n, v in dx.items() if v != 0]) + ' |a ' + ' '.join(map(str,da))

            return initial + self.vw.predict(ex)

        def update(self, xraw, z, r):
            self.incorporate()

            import numpy as np

            if r > 0 and len(z) > 1:
                (x     , a     ) = xraw
                (xprime, aprime) = z[0][0]
                (xpp   , app   ) = z[1][0]
                
                if isinstance(x[0],tuple):
                    x      = defaultdict(int,x)
                    xprime = defaultdict(int,xprime)
                    xpp    = defaultdict(int,xpp)
                else:
                    x      = defaultdict(int,enumerate(x))
                    xprime = defaultdict(int,enumerate(xprime))
                    xpp    = defaultdict(int,enumerate(xpp))

                keys = set(list(x.keys()) + list(xprime.keys()) + list(xpp.keys()))

                dx = { key:x[key]-xprime[key] for key in keys }
                da = np.hstack(a) - np.hstack(aprime)

                dxp = { key:x[key]-xpp[key] for key in keys }
                dap  = np.hstack(a) - np.hstack(app)

                if a == aprime:
                    dxa_dot_dxa = sum([2*dx[key]*dx[key] for key in keys])
                else:
                    dxa_dot_dxa = sum([dx[key]*dx[key] for key in keys]) + sum([x[key]*x[key] for key in keys]) + sum([xprime[key]*xprime[key] for key in keys])

                if a == app:
                    dxap_dot_dxap = sum([2*dxp[key]*dxp[key] for key in keys])
                else:
                    dxap_dot_dxap = sum([dxp[key]*dxp[key] for key in keys]) + sum([x[key]*x[key] for key in keys]) + sum([xpp[key]*xpp[key] for key in keys])

                initial = 0.01 * (dxa_dot_dxa - dxap_dot_dxap)

                ex = f'1 {r} {initial} |x ' + ' '.join([f'{k+1}:{dx[k]**2-dxp[k]**2}' for k in keys if dx[k]**2-dxp[k]**2 != 0 ]) + ' |a ' + ' '.join(map(str,da-dap))
                #self.vw.learn(ex)

    @staticmethod
    def routerFactory():
        return MemorizedLearner_1.LogisticModel(eta0=1e-2)

    def __init__(self, epsilon: float, max_memories: int = 1000) -> None:

        self._epsilon      = epsilon
        self._max_memories = max_memories        
        self._random       = random.Random(31337)
        self._probs        = {}
        self._update       = {}
        self._i            = 0

    def init(self):
        scorer      = MemorizedLearner_1.MarkLearnedEuclideanDistance()
        randomState = random.Random(45)
        ords        = random.Random(2112)
        self._mem   = CMT(MemorizedLearner_1.routerFactory, scorer, alpha=0.25, c=40, d=1, randomState=randomState, optimizedDeleteRandomState=ords, maxMemories=self._max_memories)

    @property
    def family(self) -> str:
        return "CMT_1"

    @property
    def params(self) -> Dict[str,Any]:
        return {'e':self._epsilon, 'm': self._max_memories, 'b': bits}

    def predict(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        predict_start = time.time()
        self._i += 1

        (greedy_r, greedy_a) = -math.inf, actions[0]

        for action in actions:
            x = self.flat(context,action)
            (_, z) = self._mem.query(x, 1, 0)
            if len(z) > 0 and z[0][1] > greedy_r:
                (greedy_r, greedy_a) = (z[0][1], action)

        ga = actions.index(greedy_a)
        minp = self._epsilon / len(actions)

        if logn and self._i % logn == 0:
           print(f"{self._i}. prediction time {round(time.time()-predict_start, 2)}")

        if self._random.random() < self._epsilon:
            ra = self._random.randint(0,len(actions)-1)
            p = 1.0 - self._epsilon + minp if ra == ga else minp
            self._probs[key] = (p, minp)
            return [ float(i == ra) for i in range(len(actions)) ]
        else:
            p = 1.0 - self._epsilon + minp
            self._probs[key] = (p, minp)
            return [ float(i == ga) for i in range(len(actions)) ]

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""

        learn_start = time.time()

        x = self.flat(context,action)

        #this reduces dependencies and simplifies code
        #but requires an extra query. If performance is
        #a problem we could use the `key` param to store
        #the result of this query in `choose` to use here
        (u,z) = self._mem.query(x, k=2, epsilon=1)

        (p, minp) = self._probs.pop(key)

        if len(z) > 0:
            megalr = 0.1
            newval = (1.0 - megalr) * z[0][1] + megalr * reward
            self._mem.updateomega(z[0][0], newval)

            self._mem.update(u, x, z, 1-(newval - reward)**2)

        # We skip for now. Alternatively we could
        # consider blending repeat contexts in the future
        if x in self._mem.leafbykey:
            self._mem.delete(x)
        self._mem.insert(x, reward)

        if logn and self._i % logn == 0:
            print(f"{self._i}. learn time {round(time.time()-learn_start, 2)}")

    def flat(self, context, action):
        if isinstance(context,dict):
            return (tuple(context.items()), action)
        else:
            return (context, action)

class ResidualLearner_1:
    def __init__(self, epsilon: float, max_memories: int):
        self._epsilon = epsilon
        self._max_memories = max_memories
        self._random = random.Random(0xdeadbeef)
        self._probs = {}
        self._i = 0
        
        self.memory = MemorizedLearner_1(0.0, self._max_memories)

    def init(self):
        from vowpalwabbit import pyvw

        #with open(devnull, 'w') as f, tools.redirect_stderr(f):
        self.vw = pyvw.vw(f'--quiet -b {bits} --cb_adf -q sa --cubic ssa --ignore_linear s')
        
        self.memory.init()

    @property
    def family(self) -> str:
        return "CMT_Residual_1"

    @property
    def params(self) -> Dict[str,Any]:
        return {'e':self._epsilon, 'm': self._max_memories, 'b': bits}

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

        exstr = self.toadf(context, actions)
        predict = self.vw.predict(exstr)
        deltas = []

        for n, action in enumerate(actions):
            mq = self.memory.flat(context, action)
            (_, z) = self.memory._mem.query(mq, 1, 0)
            deltas.append(z[0][1] if len(z) > 0 else 0)

        ga = min(((p + dp, n)
                 for p, dp, n in zip(predict, deltas, range(len(actions))))
                )[1]
        minp = self._epsilon / len(actions)

        if self._random.random() < self._epsilon:
            ra = self._random.randint(0, len(actions)-1)
            p = 1.0 - self._epsilon + minp if ra == ga else minp
            self._probs[key] = (p, minp, ra, predict[ra], actions)
            prediction = [ float(i == ra) for i in range(len(actions)) ]
        else:
            p = 1.0 - self._epsilon + minp
            self._probs[key] = (p, minp, ga, predict[ga], actions)
            prediction = [ float(i==ga) for i in range(len(actions))]

        if logn and self._i % logn == 0:
            print(f"{self._i}. prediction time {round(time.time()-predict_start, 2)}")

        return prediction


    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""

        learn_start = time.time()
        
        (prob, minp, aind, prd_loss, actions) = self._probs.pop(key)

        obs_loss  = -reward
        obs_resid = obs_loss-prd_loss
        
        exstr = self.toadf(context, actions, (aind, obs_loss, prob))
        self.vw.learn(exstr)

        x = self.memory.flat(context, action)
        (u, z) = self.memory._mem.query(x, k=2, epsilon=1)

        if len(z) > 0:
            
            smooth = 0.1
            cmt_resid = (1.0 - smooth) * z[0][1] + smooth * obs_resid
            
            self.memory._mem.updateomega(z[0][0], cmt_resid)

            deltarvw    = sorted([-1, obs_resid                     , 1])[1]
            deltarcombo = sorted([-1, obs_resid-cmt_resid           , 1])[1]
            rupdate     = sorted([0 , abs(deltarvw)-abs(deltarcombo)   ])[1]

            #supervised component, aka how valuable was u,z for query x
            self.memory._mem.update(u, x, z, rupdate)

        # replicate duplicates for now.  TODO: update memories
        if x in self.memory._mem.leafbykey:
            self.memory._mem.delete(x)
        
        self.memory._mem.insert(x, obs_resid)

        if logn and self._i % logn == 0:
            print(f"{self._i}. learn time {round(time.time()-learn_start, 2)}")

class ResidualLearner_2:
    def __init__(self, epsilon: float, max_memories: int):
        self._epsilon = epsilon
        self._max_memories = max_memories
        self._random = random.Random(0xdeadbeef)
        self._probs = {}
        self._i     = 0

        self.memory = MemorizedLearner_1(0.0, self._max_memories)

    def init(self):
        from os import devnull
        from coba import tools
        from vowpalwabbit import pyvw

        #with open(devnull, 'w') as f, tools.redirect_stderr(f):
        self.vw = pyvw.vw(f'--quiet -b {bits} --cb_adf -q sa --cubic ssa --ignore_linear s')
        
        self.memory.init()

    @property
    def family(self) -> str:
        return "CMT_Residual_2"

    @property
    def params(self) -> Dict[str,Any]:
        return {'e':self._epsilon, 'm': self._max_memories, 'b': bits}

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

        exstr = self.toadf(context, actions)
        predict = self.vw.predict(exstr)
        deltas = []

        for n, action in enumerate(actions):
            mq = self.memory.flat(context, action)
            (_, z) = self.memory._mem.query(mq, 1, 0)
            deltas.append(z[0][1] if len(z) > 0 else 0)

        ga = min(((p + dp, n)
                 for p, dp, n in zip(predict, deltas, range(len(actions))))
                )[1]
        minp = self._epsilon / len(actions)

        if logn and self._i % logn == 0:
            print(f"{self._i}. prediction time {round(time.time()-predict_start, 2)}")

        if self._random.random() < self._epsilon:
            ra = self._random.randint(0, len(actions)-1)
            p = 1.0 - self._epsilon + minp if ra == ga else minp
            self._probs[key] = (p, minp, ra, predict[ra], actions)
            return [ float(i == ra) for i in range(len(actions)) ]
        else:
            p = 1.0 - self._epsilon + minp
            self._probs[key] = (p, minp, ga, predict[ga], actions)
            return [ float(i==ga) for i in range(len(actions))]

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""

        learn_start = time.time()

        (prob, minp, aind, prd_loss, actions) = self._probs.pop(key)

        obs_loss  = -reward
        obs_resid = obs_loss-prd_loss
        
        exstr = self.toadf(context, actions, (aind, obs_loss, prob))
        self.vw.learn(exstr)

        x = self.memory.flat(context, action)
        (u, z) = self.memory._mem.query(x, k=2, epsilon=1)

        if len(z) > 0:

            smooth = 0.1
            cmt_resid = (1.0 - smooth) * z[0][1] + smooth * obs_resid

            self.memory._mem.updateomega(z[0][0], cmt_resid)

            if obs_resid == 0:
                rupdate = float(cmt_resid==0)
            else:
                rupdate = sorted([0, 1. - abs(obs_resid-cmt_resid)/abs(obs_resid)])[1]

            #supervised component, aka how valuable was u,z for query x
            self.memory._mem.update(u, x, z, rupdate)

        # replicate duplicates for now.  TODO: update memories
        if x in self.memory._mem.leafbykey:
            self.memory._mem.delete(x)

        self.memory._mem.insert(x, obs_resid)

        if logn and self._i % logn == 0:
            print(f"{self._i}. learn time {round(time.time()-learn_start, 2)}")

class JordanLogisticLearner:
    class LogisticRegressor(torch.nn.Module):
        def __init__(self, output_dim, eta0, bias=True):
            import torch

            super(MemorizedLearner_1.LogisticRegressor, self).__init__()
            self.linear = None
            self.output_dim = output_dim
            self.loss = torch.nn.CrossEntropyLoss()
            self.eta0 = eta0
            self.n = 0
            self.bias = bias

        def incorporate(self, input_dim):
            import torch
            if self.linear is None:
                self.linear = torch.nn.Linear(input_dim, self.output_dim, bias=self.bias)
                self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=self.eta0)

        def forward(self, X):
            import numpy as np
            import torch

            self.incorporate(X.shape[-1])
            return self.linear(torch.autograd.Variable(torch.from_numpy(X)))

        def predict(self, X):
            import torch

            return torch.argmax(self.forward(X), dim=1).numpy()

        def set_lr(self):
            from math import sqrt
            lr = self.eta0 / sqrt(self.n)
            for g in self.optimizer.param_groups:
                g['lr'] = lr

        def partial_fit(self, X, y, sample_weight=None, **kwargs):
            import torch

            self.incorporate(X.shape[-1])
            self.optimizer.zero_grad()
            yhat = self.forward(X)
            if sample_weight is None:
                loss = self.loss(yhat, torch.from_numpy(y))
            else:
                loss = torch.from_numpy(sample_weight) * self.loss(yhat, torch.from_numpy(y))
            loss.backward()
            self.n += X.shape[0]
            self.set_lr()
            self.optimizer.step()

    class LogisticModel:
        def __init__(self, *args, **kwargs):
            self.vw = None

        def incorporate(self):
            if self.vw is None:
                from os import devnull
                from coba import tools

                #with open(devnull, 'w') as f, tools.redirect_stderr(f):
                from vowpalwabbit import pyvw
                self.vw = pyvw.vw(f'--quiet -b {bits} --loss_function logistic -q ax --cubic axx --ignore_linear x --coin')

        def predict(self, xraw):
            self.incorporate()

            (x, a) = xraw
            ex = ' |x ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(x)]
            )  + ' |a ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(a) if v != 0]
            )
            return self.vw.predict(ex)

        def update(self, xraw, y, w):
            self.incorporate()

            (x, a) = xraw
            assert y == 1 or y == -1
            assert w >= 0
            ex = f'{y} {w} |x ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(x)]
            )  + ' |a ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(a) if v != 0]
            )

            self.vw.learn(ex)

    class FlassLogisticModel:
        def __init__(self, *args, **kwargs):
            kwargs['output_dim'] = 2
            self.model = MemorizedLearner_1.LogisticRegressor(*args, **kwargs)

        def predict(self, x):
            import numpy as np

            F = self.model.forward(X=np.array([x], dtype='float32')).detach().numpy()
            dF = F[:,1] - F[:,0]
            return -1 + 2 * dF

        def update(self, x, y, w):
            import numpy as np

            assert y == 1 or y == -1

            self.model.partial_fit(X=np.array([x], dtype='float32'),
                                   y=(1 + np.array([y], dtype='int')) // 2,
                                   sample_weight=np.array([w], dtype='float32'),
                                   classes=(0, 1))

    class LearnedEuclideanDistance:
        def __init__(self, *args, **kwargs):
            self.vw = None

        def incorporate(self):
            if self.vw is None:
                from vowpalwabbit import pyvw

                self.vw = pyvw.vw(f'--quiet -b {bits} --noconstant --loss_function logistic --coin')

        def predict(self, xraw, z):
            self.incorporate()

            import numpy as np

            (x, a) = xraw

            (xprimeraw, omegaprime) = z
            (xprime, aprime) = xprimeraw

            xa = np.reshape(np.outer(x, a), -1)
            xaprime = np.reshape(np.outer(xprime, aprime), -1)

            dxa = xa - xaprime

            ex = f' |x ' + ' '.join([f'{n+1}:{v*v}' for n, v in enumerate(dxa)])
            return -0.01 * dxa.dot(dxa) + self.vw.predict(ex)

        def update(self, xraw, z, r):
            self.incorporate()

            import numpy as np

            if r == 1 and len(z) > 1 and z[0][1] != z[1][1]:
                (x, a) = xraw
                xa = np.reshape(np.outer(x, a), -1)

                (xprime, aprime) = z[0][0]
                xaprime = np.reshape(np.outer(xprime, aprime), -1)
                dxa = xa - xaprime
                initial = -0.01 * dxa.dot(dxa)
                ex = f'1 1 {initial} |x ' + ' '.join([f'{n+1}:{v*v}' for n, v in enumerate(dxa)])
                self.vw.learn(ex)

                (xprime, aprime) = z[1][0]
                xaprime = np.reshape(np.outer(xprime, aprime), -1)
                dxa = xa - xaprime
                initial = -0.01 * dxa.dot(dxa)
                ex = f'-1 1 {initial} |x ' + ' '.join([f'{n+1}:{v*v}' for n, v in enumerate(dxa)])
                self.vw.learn(ex)

    class FlassLearnedEuclideanDistance:
        def __init__(self, *args, **kwargs):
            kwargs['output_dim'] = 2
            kwargs['bias'] = False
            self.model = MemorizedLearner_1.LogisticRegressor(*args, **kwargs)

        def incorporate(self, input_dim):
            if self.model.linear is None:
                self.model.incorporate(input_dim)
                self.model.linear.weight.data[0,:].fill_(0.01 / input_dim)
                self.model.linear.weight.data[1,:].fill_(-0.01 / input_dim)

        def predict(self, x, z):
            import numpy as np

            (xprime, omegaprime) = z

            dx = np.array([x], dtype='float32')
            dx -= [xprime]
            dx *= dx

            self.incorporate(dx.shape[-1])

            F = self.model.forward(dx).detach().numpy()
            dist = F[0,1] - F[0,0]
            return dist

        def update(self, x, z, r):
            import numpy as np

            if r == 1 and len(z) > 1 and z[0][1] != z[1][1]:
                dx = np.array([ z[0][0], z[1][0] ], dtype='float32')
                dx -= [x]
                dx *= dx
                self.incorporate(dx.shape[-1])
                y = np.array([1, 0], dtype='int')
                self.model.partial_fit(X=dx,
                                       y=y,
                                       sample_weight=None, # (?)
                                       classes=(0, 1))

    class SkLinearModel:
        def __init__(self, *args, **kwargs):
            self.model = linear_model.SGDClassifier(*args, **kwargs)

        def predict(self, x):
            try:
                return self.model.predict(X=[x])[0]
            except NotFittedError:
                return 0

        def update(self, x, y, w):
            self.model.partial_fit(X=[x], y=[y], sample_weight=[w], classes=(-1,1))

    class NormalizedLinearProduct:
        def predict(self, x, z):
            (xprime, omegaprime) = z

            xa      = np.array(x)
            xprimea = np.array(xprime)

            return np.inner(xa, xprimea) / math.sqrt(np.inner(xa, xa) * np.inner(xprimea, xprimea))

        def update(self, x, y, w):
            pass

    @staticmethod
    def routerFactory():
        #return lambda: MemorizedLearner_1.SkLinearModel(loss='log', learning_rate='constant', eta0=0.1)
        return MemorizedLearner_1.LogisticModel(eta0=1e-2)

    def __init__(self, epsilon: float, lr:float, max_memories: int = 1000) -> None:

        # SkLinearModel is fast, but kinda sucks
        # NormalizedLinearProduct is competitive and fast

        #scorer        = MemorizedLearner_1.NormalizedLinearProduct()

        scorer        = MemorizedLearner_1.LearnedEuclideanDistance(eta0=1e-2)
        randomState   = random.Random(45)
        ords          = random.Random(2112)

        #self._one_hot_encoder = OneHotEncoder()

        self._lr           = lr
        self._epsilon      = epsilon
        self._mem          = CMT(MemorizedLearner_1.routerFactory, scorer, alpha=0.25, c=10, d=1, randomState=randomState, optimizedDeleteRandomState=ords, maxMemories=max_memories)
        self._update       = {}
        self._max_memories = max_memories
        self.parametrized_model = []
        self.lf = nn.BCELoss()
        
    @property
    def family(self) -> str:
        return "jordan_logistic"

    @property
    def params(self) -> Dict[str,Any]:
        return {'e':self._epsilon, 'm': self._max_memories, 'l': self._lr, 'b': bits}

    def predict(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        #if not self._one_hot_encoder.is_fit:
        #    self._one_hot_encoder = self._one_hot_encoder.fit(actions)
        if random.random() < self._epsilon:
            ra = random.randint(0,len(actions)-1)
            return [ float(i == ra) for i in range(len(actions)) ]
        else:
            (greedy_r, greedy_a) = -math.inf, actions[0]

            for action in actions:
                x = self.flat(context, action)
                xcat = torch.Tensor(np.concatenate(x))
                param = False
                if len(self.parametrized_model) > 0:
                    output = self.parametrized_model(xcat)
                    param = True
                (_, z) = self._mem.query(x, 1, 0)
                if len(z) > 0: pred = z[0][1]
                if len(z) > 0 and param:
                    conf = 2 * torch.abs(output - 0.5)
                    scale = self.scale(conf)
                    pred = output.item()
                    #pred = output
                if len(z) > 0 and pred > greedy_r:
                    (greedy_r, greedy_a) = (z[0][1], action)
            
            ga = actions.index(greedy_a)
            return [ float(i == ga) for i in range(len(actions)) ]

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""
         
        #make this identical to Jordan's other algorithm when testing
        #1 and 1 seems to work well, at least compared to .1/.1

        x = self.flat(context, action)
        xcat = np.concatenate(x)
        if len(self.parametrized_model) == 0: 
            self.parametrized_model = nn.Sequential(nn.Linear(len(xcat), 1), nn.Sigmoid())
            #self.parametrized_model = nn.Sequential(nn.Linear(len(xcat), 10), nn.ReLU(), nn.Linear(10,1), nn.Sigmoid())
            self.scale = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
            # Mark, this controls the learning rate
            self.opt = optim.Adam(self.parametrized_model.parameters(), lr=self._lr)
            # This optimizer doesn't do anything
            self.opt_scale = optim.Adam(self.scale.parameters(), lr=self._lr)


        #this reduces dependencies and simplifies code
        #but requires an extra query. If performance is
        #a problem we could use the `key` param to store
        #the result of this query in `choose` to use here
        (u,z) = self._mem.query(x, k=1, epsilon=1)
        if len(z) > 0:
            self._mem.update(u, x, z, (1 -(z[0][1] - reward)**2))
            output = self.parametrized_model(torch.Tensor(xcat))
            conf = 2 * torch.abs(output - 0.5)
            scale = self.scale(conf)
            prediction = output
            lo = self.lf(prediction, torch.Tensor([reward]))
            lo.backward()
            self.opt.step()
            self.opt.zero_grad()
            self.opt_scale.zero_grad()


        # We skip for now. Alternatively we could
        # consider blending repeat contexts in the future
        if x in self._mem.leafbykey:
            self._mem.delete(x)
        self._mem.insert(x, reward)

    def flat(self, context,action):
        #if not isinstance(context,tuple): context = (context,)
        #one_hot_action = tuple(self._one_hot_encoder.encode([action])[0])
        #contextaction = tuple(np.reshape(np.outer(context, one_hot_action),-1))
        #return context + one_hot_action + contextaction
        return (context, action)

class JordanVowpalLearner:
    class LogisticRegressor(torch.nn.Module):
        def __init__(self, output_dim, eta0, bias=True):
            import torch

            super(MemorizedLearner_1.LogisticRegressor, self).__init__()
            self.linear = None
            self.output_dim = output_dim
            self.loss = torch.nn.CrossEntropyLoss()
            self.eta0 = eta0
            self.n = 0
            self.bias = bias

        def incorporate(self, input_dim):
            import torch
            if self.linear is None:
                self.linear = torch.nn.Linear(input_dim, self.output_dim, bias=self.bias)
                self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=self.eta0)

        def forward(self, X):
            import numpy as np
            import torch

            self.incorporate(X.shape[-1])
            return self.linear(torch.autograd.Variable(torch.from_numpy(X)))

        def predict(self, X):
            import torch

            return torch.argmax(self.forward(X), dim=1).numpy()

        def set_lr(self):
            from math import sqrt
            lr = self.eta0 / sqrt(self.n)
            for g in self.optimizer.param_groups:
                g['lr'] = lr

        def partial_fit(self, X, y, sample_weight=None, **kwargs):
            import torch

            self.incorporate(X.shape[-1])
            self.optimizer.zero_grad()
            yhat = self.forward(X)
            if sample_weight is None:
                loss = self.loss(yhat, torch.from_numpy(y))
            else:
                loss = torch.from_numpy(sample_weight) * self.loss(yhat, torch.from_numpy(y))
            loss.backward()
            self.n += X.shape[0]
            self.set_lr()
            self.optimizer.step()

    class LogisticModel:
        def __init__(self, *args, **kwargs):
            self.vw = None

        def incorporate(self):
            if self.vw is None:
                from os import devnull
                from coba import execution

                #with open(devnull, 'w') as f, execution.redirect_stderr(f):
                from vowpalwabbit import pyvw
                self.vw = pyvw.vw(f'--quiet -b {bits} --loss_function logistic -q ax --cubic axx --ignore_linear x --coin')

        def predict(self, xraw):
            self.incorporate()

            (x, a) = xraw
            ex = ' |x ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(x)]
            )  + ' |a ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(a) if v != 0]
            )
            return self.vw.predict(ex)

        def update(self, xraw, y, w):
            self.incorporate()

            (x, a) = xraw
            assert y == 1 or y == -1
            assert w >= 0
            ex = f'{y} {w} |x ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(x)]
            )  + ' |a ' + ' '.join(
                [f'{n+1}:{v}' for n, v in enumerate(a) if v != 0]
            )

            self.vw.learn(ex)

    class FlassLogisticModel:
        def __init__(self, *args, **kwargs):
            kwargs['output_dim'] = 2
            self.model = MemorizedLearner_1.LogisticRegressor(*args, **kwargs)

        def predict(self, x):
            import numpy as np

            F = self.model.forward(X=np.array([x], dtype='float32')).detach().numpy()
            dF = F[:,1] - F[:,0]
            return -1 + 2 * dF

        def update(self, x, y, w):
            import numpy as np

            assert y == 1 or y == -1

            self.model.partial_fit(X=np.array([x], dtype='float32'),
                                   y=(1 + np.array([y], dtype='int')) // 2,
                                   sample_weight=np.array([w], dtype='float32'),
                                   classes=(0, 1))

    class LearnedEuclideanDistance:
        def __init__(self, *args, **kwargs):
            self.vw = None

        def incorporate(self):
            if self.vw is None:
                from vowpalwabbit import pyvw

                self.vw = pyvw.vw(f'--quiet -b {bits} --noconstant --loss_function logistic --coin')

        def predict(self, xraw, z):
            self.incorporate()

            import numpy as np

            (x, a) = xraw

            (xprimeraw, omegaprime) = z
            (xprime, aprime) = xprimeraw

            xa = np.reshape(np.outer(x, a), -1)
            xaprime = np.reshape(np.outer(xprime, aprime), -1)

            dxa = xa - xaprime

            ex = f' |x ' + ' '.join([f'{n+1}:{v*v}' for n, v in enumerate(dxa)])
            return -0.01 * dxa.dot(dxa) + self.vw.predict(ex)

        def update(self, xraw, z, r):
            self.incorporate()

            import numpy as np

            if r == 1 and len(z) > 1 and z[0][1] != z[1][1]:
                (x, a) = xraw
                xa = np.reshape(np.outer(x, a), -1)

                (xprime, aprime) = z[0][0]
                xaprime = np.reshape(np.outer(xprime, aprime), -1)
                dxa = xa - xaprime
                initial = -0.01 * dxa.dot(dxa)
                ex = f'1 1 {initial} |x ' + ' '.join([f'{n+1}:{v*v}' for n, v in enumerate(dxa)])
                self.vw.learn(ex)

                (xprime, aprime) = z[1][0]
                xaprime = np.reshape(np.outer(xprime, aprime), -1)
                dxa = xa - xaprime
                initial = -0.01 * dxa.dot(dxa)
                ex = f'-1 1 {initial} |x ' + ' '.join([f'{n+1}:{v*v}' for n, v in enumerate(dxa)])
                self.vw.learn(ex)

    class FlassLearnedEuclideanDistance:
        def __init__(self, *args, **kwargs):
            kwargs['output_dim'] = 2
            kwargs['bias'] = False
            self.model = MemorizedLearner_1.LogisticRegressor(*args, **kwargs)

        def incorporate(self, input_dim):
            if self.model.linear is None:
                self.model.incorporate(input_dim)
                self.model.linear.weight.data[0,:].fill_(0.01 / input_dim)
                self.model.linear.weight.data[1,:].fill_(-0.01 / input_dim)

        def predict(self, x, z):
            import numpy as np

            (xprime, omegaprime) = z

            dx = np.array([x], dtype='float32')
            dx -= [xprime]
            dx *= dx

            self.incorporate(dx.shape[-1])

            F = self.model.forward(dx).detach().numpy()
            dist = F[0,1] - F[0,0]
            return dist

        def update(self, x, z, r):
            import numpy as np

            if r == 1 and len(z) > 1 and z[0][1] != z[1][1]:
                dx = np.array([ z[0][0], z[1][0] ], dtype='float32')
                dx -= [x]
                dx *= dx
                self.incorporate(dx.shape[-1])
                y = np.array([1, 0], dtype='int')
                self.model.partial_fit(X=dx,
                                       y=y,
                                       sample_weight=None, # (?)
                                       classes=(0, 1))

    class SkLinearModel:
        def __init__(self, *args, **kwargs):
            self.model = linear_model.SGDClassifier(*args, **kwargs)

        def predict(self, x):
            try:
                return self.model.predict(X=[x])[0]
            except NotFittedError:
                return 0

        def update(self, x, y, w):
            self.model.partial_fit(X=[x], y=[y], sample_weight=[w], classes=(-1,1))

    class NormalizedLinearProduct:
        def predict(self, x, z):
            (xprime, omegaprime) = z

            xa      = np.array(x)
            xprimea = np.array(xprime)

            return np.inner(xa, xprimea) / math.sqrt(np.inner(xa, xa) * np.inner(xprimea, xprimea))

        def update(self, x, y, w):
            pass

    @staticmethod
    def routerFactory():
        #return lambda: MemorizedLearner_1.SkLinearModel(loss='log', learning_rate='constant', eta0=0.1)
        return MemorizedLearner_1.LogisticModel(eta0=1e-2)

    def __init__(self, epsilon: float, lr:float, max_memories: int = 1000) -> None:

        # SkLinearModel is fast, but kinda sucks
        # NormalizedLinearProduct is competitive and fast

        #scorer        = MemorizedLearner_1.NormalizedLinearProduct()

        scorer        = MemorizedLearner_1.LearnedEuclideanDistance(eta0=1e-2)
        randomState   = random.Random(45)
        ords          = random.Random(2112)

        #self._one_hot_encoder = OneHotEncoder()

        self._lr           = lr
        self._epsilon      = epsilon
        self._mem          = CMT(MemorizedLearner_1.routerFactory, scorer, alpha=0.25, c=10, d=1, randomState=randomState, optimizedDeleteRandomState=ords, maxMemories=max_memories)
        self._update       = {}
        self._max_memories = max_memories
        self.parametrized_model = []
        self.lf = nn.BCELoss()

    @property
    def family(self) -> str:
        return "jordan_vw"

    @property
    def params(self) -> Dict[str,Any]:
        return {'e':self._epsilon, 'm': self._max_memories, 'l': self._lr, 'b': bits}

    def predict(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        #if not self._one_hot_encoder.is_fit:
        #    self._one_hot_encoder = self._one_hot_encoder.fit(actions)
        if random.random() < self._epsilon:
            ra = random.randint(0,len(actions)-1)
            return [ float(i == ra) for i in range(len(actions)) ]
        else:
            (greedy_r, greedy_a) = -math.inf, actions[0]

            for action in actions:
                x = self.flat(context, action)
                xcat = torch.Tensor(np.concatenate(x))
                param = False
                if len(self.parametrized_model) > 0:
                    output = self.parametrized_model(xcat)
                    param = True
                (_, z) = self._mem.query(x, 1, 0)
                if len(z) > 0: pred = z[0][1]
                if len(z) > 0 and param:
                    conf = 2 * torch.abs(output - 0.5)
                    scale = self.scale(conf)
                    pred = 0.5 * (output +  pred).item()
                    #pred = output
                if len(z) > 0 and pred > greedy_r:
                    (greedy_r, greedy_a) = (z[0][1], action)

            ga = actions.index(greedy_a)
            return [ float(i == ga) for i in range(len(actions)) ]

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float, probability:float) -> None:
        """Learn about the result of an action that was taken in a context."""
         
        x = self.flat(context, action)
        xcat = np.concatenate(x)
        if len(self.parametrized_model) == 0: 
            self.parametrized_model = nn.Sequential(nn.Linear(len(xcat), 1), nn.Sigmoid())
            self.scale = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
            #### Mark, change the learning rate below
            self.opt = optim.Adam(self.parametrized_model.parameters(), lr=self._lr)
            ## the scale isn't being used here, that was a separate experiment
            self.opt_scale = optim.Adam(self.scale.parameters(), lr=self._lr)


        #this reduces dependencies and simplifies code
        #but requires an extra query. If performance is
        #a problem we could use the `key` param to store
        #the result of this query in `choose` to use here
        (u,z) = self._mem.query(x, k=1, epsilon=1)
        if len(z) > 0:
            self._mem.update(u, x, z, (1 -(z[0][1] - reward)**2))
            output = self.parametrized_model(torch.Tensor(xcat))
            conf = 2 * torch.abs(output - 0.5)
            scale = self.scale(conf)
            prediction = 0.5 * (output +  z[0][1])
            lo = self.lf(prediction, torch.Tensor([reward]))
            lo.backward()
            self.opt.step()
            self.opt.zero_grad()
            self.opt_scale.zero_grad()


        # We skip for now. Alternatively we could
        # consider blending repeat contexts in the future
        if x in self._mem.leafbykey:
            self._mem.delete(x)
        self._mem.insert(x, reward)

    def flat(self, context,action):
        #if not isinstance(context,tuple): context = (context,)
        #one_hot_action = tuple(self._one_hot_encoder.encode([action])[0])
        #contextaction = tuple(np.reshape(np.outer(context, one_hot_action),-1))
        #return context + one_hot_action + contextaction
        return (context, action)
