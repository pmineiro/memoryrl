import random
import math

from typing import Hashable, Sequence, Dict, Any

import numpy as np
import torch

from sklearn.exceptions import NotFittedError 
from sklearn import linear_model

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
                self.right = replacement.right
                if self.right:
                    self.right.parent = self
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
                y = (1 - self.alpha) * rhat + self.alpha * (log(1e-2 + v.left.n) - log(1e-2 + v.right.n)) 
                signy = 1 if y > 0 else -1
                absy = signy * y
                v.g.update(x, signy, absy)
                
            for _ in range(self.d):
                self.__reroute()
                
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
        
        for k in self.leafbykey.keys():
            assert k in self.leafbykey[k].memories

class MemorizedLearner_1:

    class LinearModel:
        def __init__(self, *args, **kwargs):
            self.model = linear_model.SGDRegressor(*args, **kwargs)

        def predict(self, x):
            try:
                return self.model.predict(X=[x])[0]
            except NotFittedError:
                return 0
        
        def update(self, x, y, w):
            self.model.partial_fit(X=[x], y=[y], sample_weight=[w])
            
    class NormalizedLinearProduct:
        def predict(self, x, z):
            (xprime, omegaprime) = z

            xa      = np.array(x)
            xprimea = np.array(xprime)

            return np.inner(xa, xprimea) / math.sqrt(np.inner(xa, xa) * np.inner(xprimea, xprimea))

        def update(self, x, y, w):
            pass

    def __init__(self, epsilon: float, max_memories: int = 1000) -> None:

        routerFactory = lambda: MemorizedLearner_1.LinearModel()
        scorer        = MemorizedLearner_1.NormalizedLinearProduct()
        randomState   = random.Random(45)

        self._one_hot_encoder = OneHotEncoder()

        self._epsilon      = epsilon
        self._mem          = CMT(routerFactory, scorer, alpha=0.25, c=10, d=1, randomState=randomState, maxMemories=max_memories)
        self._update       = {}
        self._max_memories = max_memories

    @property
    def family(self) -> str:
        return "CMT_1"

    @property
    def params(self) -> Dict[str,Any]:
        return {'e':self._epsilon, 'm': self._max_memories}

    def choose(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> int:
        """Choose which action index to take."""

        if not self._one_hot_encoder.is_fit:
            self._one_hot_encoder = self._one_hot_encoder.fit(actions)

        if random.random() < self._epsilon:
            return random.randint(0,len(actions)-1)
        else:
            (greedy_r, greedy_a) = -math.inf, actions[0]

            for action in actions:
                x = self.flat(context,action)
                (_, z) = self._mem.query(x, 1, 0)
                if len(z) > 0 and z[0][1] > greedy_r:
                    (greedy_r, greedy_a) = (z[0][1], action)

            return actions.index(greedy_a)

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float) -> None:
        """Learn about the result of an action that was taken in a context."""
        
        x = self.flat(context,action)

        #this reduces dependencies and simplifies code
        #but requires an extra query. If performance is
        #a problem we could use the `key` param to store
        #the result of this query in `choose` to use here
        (u,z) = self._mem.query(x, k=1, epsilon=1)

        if len(z) > 0:
            self._mem.update(u, x, z, reward)

        # We skip for now. Alternatively we could
        # consider blending repeat contexts in the future
        if x not in self._mem.leafbykey:
            self._mem.insert(x, reward)

    def flat(self, context,action):

        if not isinstance(context,tuple): context = (context,)
        if not isinstance(action ,tuple): action  = (action,)

        one_hot_action = tuple(self._one_hot_encoder.encode([action[0]])[0])

        return context + one_hot_action + tuple(np.reshape(np.outer(context, one_hot_action),-1))

class MemorizedLearner_2:

    class LogisticRegressor(torch.nn.Module):        
        def __init__(self, input_dim, output_dim, eta0):
            
            super(MemorizedLearner_2.LogisticRegressor, self).__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)
            self.loss = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=eta0)
            self.eta0 = eta0
            self.n = 0
            
        def forward(self, X):

            return self.linear(torch.autograd.Variable(torch.from_numpy(X)))
        
        def predict(self, X):            
            return torch.argmax(self.forward(X), dim=1).numpy()
        
        def set_lr(self):
            from math import sqrt
            lr = self.eta0 / sqrt(self.n)
            for g in self.optimizer.param_groups:
                g['lr'] = lr

        def partial_fit(self, X, y, sample_weight=None, **kwargs):
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
            kwargs['output_dim'] = 2
            self.model = MemorizedLearner_2.LogisticRegressor(*args, **kwargs)
            
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
            kwargs['output_dim'] = 2
            self.model = MemorizedLearner_2.LogisticRegressor(*args, **kwargs)
            self.model.linear.weight.data[0,:].fill_(0.01 / kwargs['input_dim'])
            self.model.linear.weight.data[1,:].fill_(-0.01 / kwargs['input_dim'])
            self.model.linear.bias.data.fill_(0.0)
            self.model.linear.bias.requires_grad = False
        
        def predict(self, x, z):
            import numpy as np
            
            (xprime, omegaprime) = z
            
            dx = np.array([x], dtype='float32')
            dx -= [xprime]
            dx *= dx
            
            F = self.model.forward(dx).detach().numpy()
            dist = F[0,1] - F[0,0]
            return dist
        
        def update(self, x, z, r):
            import numpy as np
            
            if r == 1 and len(z) > 1 and z[0][1] != z[1][1]:
                dx = np.array([ z[0][0], z[1][0] ], dtype='float32')
                dx -= [x]
                dx *= dx
                y = np.array([1, 0], dtype='int')    
                self.model.partial_fit(X=dx,
                                       y=y,
                                       sample_weight=None, # (?)
                                       classes=(0, 1))

    def __init__(self, epsilon: float, max_memories: int = 1000) -> None:

        routerFactory = lambda: MemorizedLearner_1.LinearModel()
        scorer        = MemorizedLearner_1.NormalizedLinearProduct()
        randomState   = random.Random(45)

        self._epsilon      = epsilon
        self._mem          = CMT(routerFactory, scorer, alpha=0.25, c=10, d=1, randomState=randomState, maxMemories=max_memories)
        self._update       = {}
        self._max_memories = max_memories

    @property
    def family(self) -> str:
        return "CMT_1"

    @property
    def params(self) -> Dict[str,Any]:
        return {'e':self._epsilon, 'm': self._max_memories}

    def choose(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> int:
        """Choose which action index to take."""

        if random.random() < self._epsilon:
            return random.randint(0,len(actions)-1)
        else:
            (greedy_r, greedy_a) = -math.inf, actions[0]

            for action in actions:
                x = MemorizedLearner_1.flat(context,action)
                (_, z) = self._mem.query(x, 1, 0)
                if len(z) > 0 and z[0][1] > greedy_r:
                    (greedy_r, greedy_a) = (z[0][1], action)

            return actions.index(greedy_a)

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float) -> None:
        """Learn about the result of an action that was taken in a context."""
        
        x = MemorizedLearner_1.flat(context,action)

        #this reduces dependencies and simplifies code
        #but requires an extra query. If performance is
        #a problem we could use the `key` param to store
        #the result of this query in `choose` to use here
        (u,z) = self._mem.query(x, 2, 1)

        if len(z) > 0:
            1 if z[0][1] == actual else 0
            self._mem.update(u, x, z, -(z[0][1]-reward)**2)
        else:
            self._mem.insert(x, reward)

    @staticmethod
    def flat(context,action):
        if not isinstance(context,tuple): context = (context,)
        if not isinstance(action ,tuple): action  = (action,)

        return context+action