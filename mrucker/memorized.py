import random
import math

from typing import Hashable, Sequence, Dict, Any

import numpy as np

from coba.preprocessing import OneHotEncoder

class CMT:
    class Node:
        def __init__(self, parent, left=None, right=None, g=None):
            self.parent = parent
            self.isLeaf = left is None
            self.n = 0
            self.memories = {}
            self.left = left
            self.right = right
            self.g = g
            self.depth = 0 if parent is None else (1 + self.parent.depth)
            #assert self.depth < 10, f'wtf {self.depth}'

        def makeInternal(self, g):
            assert self.isLeaf

            self.isLeaf = False
            self.left = CMT.Node(parent=self)
            self.right = CMT.Node(parent=self)
            self.n = 0
            self.g = g

            mem = self.memories
            self.memories = {}

            return mem

        def replaceNode(self, replacement):
            if self is not replacement:
                self.isLeaf = replacement.isLeaf
                self.n = replacement.n
                self.memories = replacement.memories
                self.left = replacement.left
                if self.left:
                    self.left.parent = self
                    self.left.depth = 1 + self.depth
                self.right = replacement.right
                if self.right:
                    self.right.parent = self
                    self.right.depth = 1 + self.depth
                self.g = replacement.g

        def topk(self, x, k, f):
            assert self.isLeaf
            return [ z for _, z in zip(range(k),
                                       sorted(self.memories.items(),
                                              key=lambda z: f.predict(x, z),
                                              reverse=True
                                             )
                                      )
                   ]

        def randk(self, k, randomState):
            assert self.isLeaf
            return [ z[1] for _, z in zip(range(k),
                                          sorted( (randomState.uniform(0, 1), m) for m in self.memories.items()
                                                )
                                      )
                   ]

    class Path:
        def __init__(self, nodes, leaf):
            self.nodes = nodes
            self.leaf = leaf

    class LRU:
        def __init__(self):
            self.entries = []
            self.entry_finder = {}
            self.n = 0

        def add(self, x):
            from heapq import heappush

            assert x not in self.entry_finder

            entry = (self.n, x)
            self.entry_finder[x] = self.n
            heappush(self.entries, entry)
            self.n += 1

        def __len__(self):
            return len(self.entry_finder)

        def __contains__(self, x):
            return x in self.entry_finder

        def peek(self):
            from heapq import heappop

            while self.entry_finder.get(self.entries[0][1], -1) != self.entries[0][0]:
                heappop(self.entries)

            return self.entries[0][1]

        def remove(self, x):
            self.entry_finder.pop(x)

    def __init__(self, routerFactory, scorer, alpha, c, d, randomState, maxMemories=None, optimizedDeleteRandomState=None):
        self.routerFactory = routerFactory
        self.f = scorer
        self.alpha = alpha
        self.c = c
        self.d = d
        self.leafbykey = {}
        self.root = CMT.Node(None)
        self.randomState = randomState
        self.allkeys = []
        self.allkeysindex = {}
        self.maxMemories = maxMemories
        self.odrs = optimizedDeleteRandomState
        self.keyslru = CMT.LRU()
        self.rerouting = False
        self.splitting = False

    def nodeForeach(self, f, node=None):
        if node is None:
            node = self.root

        f(node)
        if node.left:
            self.nodeForeach(f, node.left)
        if node.right:
            self.nodeForeach(f, node.right)

    def __path(self, x, v):
        nodes = []
        while not v.isLeaf:
            a = v.right if v.g.predict(x) > 0 else v.left
            nodes.append(v)
            v = a

        return CMT.Path(nodes, v)

    def query(self, x, k, epsilon):
        path = self.__path(x, self.root)
        q = self.randomState.uniform(0, 1)
        if q >= epsilon:
            return (None, path.leaf.topk(x, k, self.f))
        else:
            i = self.randomState.randint(0, len(path.nodes))
            if i < len(path.nodes):
                a = self.randomState.choice( (path.nodes[i].left, path.nodes[i].right) )
                l = self.__path(x, a).leaf
                return ((path.nodes[i], a, 1/2), l.topk(x, k, self.f))
            else:
                return ((path.leaf, None, None), path.leaf.randk(k, self.randomState))

    def update(self, u, x, z, r):
        assert 0 <= r <= 1
        if u is None:
            pass
        else:
            (v, a, p) = u
            if v.isLeaf:
                self.f.update(x, z, r)

                if self.odrs is not None and len(z) > 0:
                    q = self.odrs.uniform(0, 1)
                    if q <= r:
                        self.keyslru.remove(z[0][0])
                        self.keyslru.add(z[0][0])
            else:
                from math import log

                rhat = (r/p) * (1 if a == v.right else -1)
                B = log(1e-2 + v.left.n) - log(1e-2 + v.right.n)
                y = (1 - self.alpha) * rhat + self.alpha * B
                signy = 1 if y > 0 else -1
                absy = signy * y
                v.g.update(x, signy, absy)

            for _ in range(self.d):
                self.__reroute()

    def updateomega(self, x, newomega):
        v = self.leafbykey[x]
        assert v.isLeaf
        v.memories[x] = newomega

    def delete(self, x):
        if x not in self.allkeysindex:
            # deleting something not in the memory ...
            assert False

        ind = self.allkeysindex.pop(x)
        lastx = self.allkeys.pop()
        if ind < len(self.allkeys):
            self.allkeys[ind] = lastx
            self.allkeysindex[lastx] = ind

        if not self.rerouting:
            self.keyslru.remove(x)

        v = self.leafbykey.pop(x)

        while v is not None:
            v.n -= 1
            if v.isLeaf:
                omega = v.memories.pop(x)
            else:
                if v.n == 0:
                    other = v.parent.left if v is v.parent.right else v.parent.right
                    if other.isLeaf:
                        for xprime in other.memories.keys():
                            self.leafbykey[xprime] = v.parent

                    v.parent.replaceNode(other)
                    v = v.parent

            assert v.n >= 0
            v = v.parent

    def __insertLeaf(self, x, omega, v):
        from math import log

        assert v.isLeaf

        if x not in self.allkeysindex:
            self.allkeysindex[x] = len(self.allkeys)
            self.allkeys.append(x)

        if not self.rerouting and not self.splitting:
            self.keyslru.add(x)

        if self.splitting or v.n < self.c:
            assert x not in self.leafbykey
            self.leafbykey[x] = v
            assert x not in v.memories
            v.memories[x] = omega
            v.n += 1
            assert v.n == len(v.memories)
        else:
            self.splitting = True
            mem = v.makeInternal(g=self.routerFactory())

            while mem:
                xprime, omegaprime = mem.popitem()
                del self.leafbykey[xprime]
                self.insert(xprime, omegaprime, v)

            self.insert(x, omega, v)
            self.splitting = False

        if not self.rerouting and not self.splitting:
            daleaf = self.leafbykey[x]
            dabest = daleaf.topk(x, 2, self.f)
            if len(dabest) > 1:
                other = dabest[1] if dabest[0][0] == x else dabest[0]
                z = [(x, omega), other]
                self.f.update(x, z, 1)

    def insert(self, x, omega, v=None):
        from math import log

        if x in self.leafbykey:
            # duplicate memory ... need to merge values ...
            assert False

        if v is None:
            v = self.root

        while not v.isLeaf:
            B = log(1e-2 + v.left.n) - log(1e-2 + v.right.n)
            y = (1 - self.alpha) * v.g.predict(x) + self.alpha * B
            signy = 1 if y > 0 else -1
            v.g.update(x, signy, 1)
            v.n += 1
            v = v.right if v.g.predict(x) > 0 else v.left

        self.__insertLeaf(x, omega, v)

        if not self.rerouting and not self.splitting:
            if self.maxMemories is not None and len(self.keyslru) > self.maxMemories:
                leastuseful = self.keyslru.peek()
                self.delete(leastuseful)

            for _ in range(self.d):
                self.__reroute()

    def __reroute(self):
        x = self.randomState.choice(self.allkeys)
        omega = self.leafbykey[x].memories[x]
        self.rerouting = True
        self.delete(x)
        self.insert(x, omega)
        self.rerouting = False

        if False:
            for k in self.leafbykey.keys():
                assert k in self.leafbykey[k].memories

class MemorizedLearner_1:
    class LogisticModel:
        def __init__(self, *args, **kwargs):
            self.vw = None

        def incorporate(self):
            if self.vw is None:
                from os import devnull
                from coba import execution

                with open(devnull, 'w') as f, execution.redirect_stderr(f):
                    from vowpalwabbit import pyvw
                    self.vw = pyvw.vw('--quiet -b 16 --loss_function logistic --link=glf1 -q ax --cubic axx --coin')

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

                self.vw = pyvw.vw('--quiet -b 16 --noconstant --loss_function logistic -qxx --link=glf1 --coin')

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


    @staticmethod
    def routerFactory():
        return MemorizedLearner_1.LogisticModel(eta0=1e-2)

    def __init__(self, epsilon: float, max_memories: int = 1000) -> None:

        scorer        = MemorizedLearner_1.LearnedEuclideanDistance(eta0=1e-2)
        randomState   = random.Random(45)
        ords          = random.Random(2112)

        self._epsilon      = epsilon
        self._probs        = {}
        self._mem          = CMT(MemorizedLearner_1.routerFactory, scorer, alpha=0.25, c=40, d=1, randomState=randomState, optimizedDeleteRandomState=ords, maxMemories=max_memories)
        self._update       = {}
        self._max_memories = max_memories
        self._random = random.Random(31337)

    @property
    def family(self) -> str:
        return "CMT_1"

    @property
    def params(self) -> Dict[str,Any]:
        return {'e':self._epsilon, 'm': self._max_memories}

    def choose(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> int:
        """Choose which action index to take."""

        (greedy_r, greedy_a) = -math.inf, actions[0]

        for action in actions:
            x = self.flat(context,action)
            (_, z) = self._mem.query(x, 1, 0)
            if len(z) > 0 and z[0][1] > greedy_r:
                (greedy_r, greedy_a) = (z[0][1], action)

        ga = actions.index(greedy_a)
        minp = self._epsilon / len(actions)

        if self._random.random() < self._epsilon:
            ra = self._random.randint(0,len(actions)-1)
            p = 1.0 - self._epsilon + minp if ra == ga else minp
            self._probs[key] = (p, minp)
            return ra
        else:
            p = 1.0 - self._epsilon + minp
            self._probs[key] = (p, minp)
            return ga

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float) -> None:
        """Learn about the result of an action that was taken in a context."""

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

            self._mem.update(u, x, z, (1 -(newval - reward)**2))

        # We skip for now. Alternatively we could
        # consider blending repeat contexts in the future
        if x in self._mem.leafbykey:
            self._mem.delete(x)
        self._mem.insert(x, reward)

    def flat(self, context, action):
        return (context, action)

class ResidualLearner_1:
    def __init__(self, epsilon: float, max_memories: int):
        from os import devnull
        from coba import execution

        with open(devnull, 'w') as f, execution.redirect_stderr(f):
            from vowpalwabbit import pyvw
            self.vw = pyvw.vw(f'--quiet --cb_adf -q sa --cubic ssa --ignore_linear s')
        self.memory = MemorizedLearner_1(0.0, max_memories)
        self._epsilon = epsilon
        self._max_memories = max_memories
        self._random = random.Random(0xdeadbeef)
        self._probs = {}

    @property
    def family(self) -> str:
        return "CMT_Residual"

    @property
    def params(self) -> Dict[str,Any]:
        return {'e':self._epsilon, 'm': self._max_memories}

    def toadf(self, context, actions, label=None):
        assert type(context) is tuple, context

        return '\n'.join([
          'shared |s ' + ' '.join([ f'{k+1}:{v}' for k, v in enumerate(context) ]),
          ] + [
            f'{dacost} |a ' + ' '.join([ f'{k+1}:{v}' for k, v in enumerate(a) if v != 0 ])
            for n, a in enumerate(actions)
            for dacost in ((f'0:{label[1]}:{label[2]}' if label is not None and n == label[0] else ''),)
        ])

    def choose(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> int:
        """Choose which action index to take."""

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
            return ra
        else:
            p = 1.0 - self._epsilon + minp
            self._probs[key] = (p, minp, ga, predict[ga], actions)
            return ga


    def learn(self, key: int, context: Hashable, action: Hashable, reward: float) -> None:
        """Learn about the result of an action that was taken in a context."""

        (p, minp, aind, ascore, actions) = self._probs.pop(key)
        exstr = self.toadf(context, actions, (aind, -reward, p))
        self.vw.learn(exstr)

        x = self.memory.flat(context, action)
        (u, z) = self.memory._mem.query(x, k=2, epsilon=1)

        if len(z) > 0:
            megalr = 0.1
            newval = (1.0 - megalr) * z[0][1] + megalr * (-reward - ascore)
            self.memory._mem.updateomega(z[0][0], newval)

            deltarvw = max(-1, min(1, ascore + reward))
            deltarcombo = max(-1, min(1, ascore + newval + reward))
            rupdate = max(0, abs(deltarvw) - abs(deltarcombo))

            self.memory._mem.update(u, x, z, rupdate)

        # replicate duplicates for now.  TODO: update memories
        if x in self.memory._mem.leafbykey:
            self.memory._mem.delete(x)
        self.memory._mem.insert(x, -reward - ascore)
