import random
import math

from typing import Hashable, Sequence

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
            self.entry_finder = set()
            self.n = 0
        
        def add(self, x):
            from heapq import heappush
            
            assert x not in self.entry_finder
            
            entry = (self.n, x)
            self.entry_finder.add(x)
            heappush(self.entries, entry)
            self.n += 1
            
        def __len__(self):
            return len(self.entry_finder)
        
        def __contains__(self, x):
            return x in self.entry_finder
        
        def peek(self):
            from heapq import heappop
            
            while self.entries[0][1] not in self.entry_finder:
                heappop(self.entries)
                
            return self.entries[0][1]
        
        def remove(self, x):
            self.entry_finder.remove(x)
    
    def __init__(self, routerFactory, scorer, alpha, c, d, randomState, maxMemories=None):
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
        if u is None:
            pass
        else:
            (v, a, p) = u
            if v.isLeaf:
                self.f.update(x, z, r)
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
                oldest = self.keyslru.peek()
                self.delete(oldest)

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

class CMTTests:
    class LinearModel:
        def __init__(self, *args, **kwargs):
            from sklearn import linear_model
            
            self.model = linear_model.SGDRegressor(*args, **kwargs)
            
        def predict(self, x):
            from sklearn.exceptions import NotFittedError 
            try:
                return self.model.predict(X=[x])[0]
            except NotFittedError:
                return 0
        
        def update(self, x, y, w):
            self.model.partial_fit(X=[x], y=[y], sample_weight=[w])
            
    class NormalizedLinearProduct:
        def __init__(self):
            pass
        
        def predict(self, x, z):
            import numpy as np
            from math import sqrt
            
            (xprime, omegaprime) = z
            
            xa = np.array(x)
            xprimea = np.array(xprime)
                        
            return np.inner(xa, xprimea) / sqrt(np.inner(xa, xa) * np.inner(xprimea, xprimea))
        
        def update(self, x, y, w):
            pass
 
    @staticmethod
    def displaynode(node, indent):
        if node is not None:
            from pprint import pformat
            print(indent, pformat((node, node.__dict__)))
            CMTTests.displaynode(node.left, indent + "*")
            CMTTests.displaynode(node.right, indent + "*")

    @staticmethod
    def displaytree(cmt):
        CMTTests.displaynode(cmt.root, indent="")

    @staticmethod
    def structureValid():
        import random
        
        routerFactory = lambda: CMTTests.LinearModel()
        scorer = CMTTests.NormalizedLinearProduct()
        randomState = random.Random()
        randomState.seed(2112)
        cmt = CMT(routerFactory=routerFactory, scorer=scorer, alpha=0.5, c=10, d=0, randomState=randomState)

        def checkNodeInvariants(node):
            assert node.parent is None or node.parent.left is node or node.parent.right is node
            assert node.left is None or node.n == node.left.n + node.right.n
            assert node.left is None or node.left.parent is node
            assert node.right is None or node.right.parent is node
            assert node.left is not None or node.n == len(node.memories)
    
        stuff = {}
        
        for _ in range(200):
            try:
                if stuff and randomState.uniform(0, 1) < 0.1:
                    # delete
                    x, omega = stuff.popitem()
                    cmt.delete(x)
                elif stuff and randomState.uniform(0, 1) < 0.1:
                    # query/update
                    somex = randomState.choice(list(stuff.keys()))
                    u, z = cmt.query(somex, 1, 0.1)
                    cmt.update(u, somex, z, randomState.uniform(0, 1))
                else:
                    # insert
                    x = tuple([ randomState.uniform(0, 1) for _ in range(3)])
                    omega = randomState.uniform(0, 1)
                    cmt.insert(x, omega)
                    stuff[x] = omega

                assert cmt.root.n == len(stuff)
                assert cmt.root.n == len(cmt.leafbykey)
                assert cmt.root.n == len(cmt.allkeys)
                assert cmt.root.n == len(cmt.allkeysindex)
                
                for z in stuff.keys():
                    assert z in cmt.leafbykey[z].memories
                    assert z in cmt.allkeysindex
                    assert cmt.allkeys[cmt.allkeysindex[z]] is z
                cmt.nodeForeach(checkNodeInvariants)
            except:
                print("--------------")
                CMTTests.displaytree(cmt)
                print("--------------")
                raise
                
        print('structureValid test pass')           
                       
    @staticmethod
    def selfconsistent():
        import random
        
        routerFactory = lambda: CMTTests.LinearModel()
        scorer = CMTTests.NormalizedLinearProduct()
        randomState = random.Random()
        randomState.seed(45)
        cmt = CMT(routerFactory=routerFactory, scorer=scorer, alpha=0.5, c=10, d=0, randomState=randomState)
        
        for _ in range(200):
            try:
                x = tuple([ randomState.uniform(0, 1) for _ in range(3)])
                omega = randomState.uniform(0, 1)

                cmt.insert(x, omega)
                u, [ (xprime, omegaprime) ] = cmt.query(x, k=1, epsilon=0)
                assert omega == omegaprime, '({}, [({}, {})]) = cmt.query({}) != {}'.format(u, xprime, omegaprime, x, omega)
            except:
                print("--------------")
                CMTTests.displaytree(cmt)
                print("--------------")
                raise
                
        print('selfconsistent test pass')
        
    @staticmethod
    def maxmemories():
        import random
        
        routerFactory = lambda: CMTTests.LinearModel()
        scorer = CMTTests.NormalizedLinearProduct()
        randomState = random.Random()
        randomState.seed(45)
        maxM = 100
        cmt = CMT(routerFactory=routerFactory, scorer=scorer, alpha=0.5, c=10, d=0, randomState=randomState, maxMemories=maxM)
        
        for _ in range(200):
            try:
                x = tuple([ randomState.uniform(0, 1) for _ in range(3)])
                omega = randomState.uniform(0, 1)

                cmt.insert(x, omega)
                assert len(cmt.leafbykey) <= maxM
            except:
                print("--------------")
                CMTTests.displaytree(cmt)
                print("--------------")
                raise
                
        print('maxmemories test pass')
       
    @staticmethod
    def all():
        CMTTests.structureValid()
        CMTTests.selfconsistent()
        CMTTests.maxmemories()

CMTTests().all()



# Basic idea:

# Estimate value of $a$ in context $x$ by stored value associated with first memory retrieved from CMT queried with $(x, a)$.
# Play $\epsilon$-greedy with greedy action being the maximum estimated value.
# Play action $a$ in context $x$ and observe reward $r$.
# Reward memory system just like a parametric direct method, i.e., using regression loss such as squared loss.
# Update the memory $((x', a'), r')$ retrieved by query $(x, a)$ using reward $-(r - r')^2$.
# Insert key $(x, a)$ with value $r$.
# Conjecture: compatible with self-consistency assuming no reward variance.
# Update reward is maximized by retrieving a memory with $r = r'$.
# Exact match response does this.
# Censorship issue: only argmax key is updated, does this matter?

class memorized_learner:

    def __init__(self, epsilon: float) -> None:
        self._epsilon = epsilon
        self._mem     = CMT()
        self._update  = {}

    def choose(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> int:
        """Choose which action index to take."""

        if random.random() < self._epsilon:
            return random.choice(actions)
        else:
            (greedy_r, greedy_a) = -math.inf, actions[0]

            for action in actions:
                (_, z) = self._mem.query((context,action), 1, 0)
                if len(z) > 0 and z[0][1] > greedy_r:
                    (greedy_r, greedy_a) = (z[0][1], action)

            return actions.index(greedy_a)

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float) -> None:
        """Learn about the result of an action that was taken in a context."""
        
        #this reduces dependencies and simplifies code
        #but requires an extra query. If performance is
        #a problem we could use the `key` param to store
        #the result of this query in `choose` to use here
        (u,z) = self._mem.query((context,action), 1, 0)

        if len(z) > 0:
            self._mem.update(u, (context,action), z, -(z[0][1]-reward)**2)
        else:
            self._mem.insert((context,action), reward)