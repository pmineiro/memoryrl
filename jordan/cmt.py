import pdb
import numpy as np

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
    
    def __init__(self, routerFactory, scorer, alpha, c, d, randomState, maxMemories=None, pruneStrat='ci'):
        self.routerFactory = routerFactory
        self.f = scorer
        self.alpha = alpha
        self.c = c
        self.d = d
        self.leafbykey = {}
        self.root = CMT.Node(None)
        self.randomState = randomState        
        self.allkeys = []
        self.pruneStrat = pruneStrat
        self.allkeysindex = {}
        self.maxMemories = maxMemories 
        self.keyslru = CMT.LRU()
        self.rerouting = False
        self.splitting = False
        self.freq = {}
        self.qCount = 0

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
            #print(len(xhat), flush=True)
            self.qCount += 1
            toReturn = path.leaf.topk(x, k, self.f)
            if len(toReturn) > 0: self.freq[toReturn[0][0]][0] += 1
            return (None, toReturn)
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
        if x not in self.freq.keys():
            self.freq[x] = [0, self.qCount]
        if not self.rerouting and not self.splitting:
            if self.maxMemories is not None and len(self.keyslru) > self.maxMemories:
                if self.pruneStrat == 'rand':
                    x = self.randomState.choice(self.allkeys)
                    self.delete(x)
                if self.pruneStrat == 'old':
                    oldest = self.keyslru.peek()
                    self.delete(oldest)
                if self.pruneStrat == 'ci':
                    keys = [k for k in self.allkeys]
                    vals = np.asarray([self.freq[k] for k in self.allkeys])
                    phat = vals[:, 0] / (self.qCount - vals[:, 1] + 1e-7) + 1e-7
                    scores = phat + 1.96 * np.sqrt(phat * (1 - phat) / (self.qCount - vals[:, 1] + 1e-7) + 1e-7)
                    self.delete(keys[np.argmin(scores)])

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

    def __rand_remove(self):
        x = self.randomState.choice(self.allkeys)
        self.delete(x)
    
    def __bci_remove(self):
        x = self.randomState.choice(self.allkeys)
        self.delete(x)
