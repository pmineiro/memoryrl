from math    import log
from heapq   import heappush, heappop
from typing  import Dict, Any, Sequence, Tuple, Hashable, Optional

from routers import Router, RouterFactory
from scorers import Scorer
from random  import Random

MemKey = Hashable
MemVal = Any
Memory = Tuple[MemKey,MemVal]

class CMT:

    class Node:
        def __init__(self, parent:'CMT.Node', rng: Random = None):
            self.parent = parent
            self.rng    = rng if rng else parent.rng

            self.memories: Dict[MemKey,MemVal] = {}

            self.n        = 0
            self.left     = None
            self.right    = None
            self.g        = None

        @property
        def is_leaf(self) -> bool:
            return self.left is None

        @property
        def depth(self):
            return 1 + self.parent.depth if self.parent else 0

        def make_internal(self, g: Router):
            assert self.is_leaf

            self.left    = CMT.Node(self)
            self.right   = CMT.Node(self)
            self.n       = 0
            self.g       = g

            mem = self.memories
            self.memories = {}

            return mem

        def replace_node(self, replacement: 'CMT.Node'):
            if self is not replacement:
                self.n        = replacement.n
                self.memories = replacement.memories
                self.left     = replacement.left
                self.right    = replacement.right
                self.g        = replacement.g

                if self.left : self.left.parent = self
                if self.right: self.right.parent = self

        def topk(self, x, k:int, f:Scorer) -> Sequence[Memory]:

            assert self.is_leaf

            keys        = list(self.memories.keys())
            scores      = dict(zip(keys,f.predict(x, keys)))
            sorted_keys = sorted(keys, key=scores.__getitem__, reverse=True) 

            return [ (key,self.memories[key]) for key in sorted_keys[0:k]]

        def randk(self, k) -> Sequence[Memory]:

            assert self.is_leaf

            memories = list(self.memories.items())
            self.rng.shuffle(memories)

            return memories[0:k]

    class Path:
        def __init__(self, nodes: Sequence['CMT.Node'], leaf: 'CMT.Node'):
            self.nodes = nodes
            self.leaf = leaf

    class LRU:
        def __init__(self):
            self.entries = []
            self.entry_finder = {}
            self.n = 0

        def add(self, x: MemKey):

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
            while self.entry_finder.get(self.entries[0][1], -1) != self.entries[0][0]:
                heappop(self.entries)

            return self.entries[0][1]

        def remove(self, x: MemKey):
            self.entry_finder.pop(x)

    def __init__(self, max_mem:int, router: RouterFactory, scorer: Scorer, c:int, d:int, alpha:float=0.25, rng:Random= Random(1337)):

        self.max_mem   = max_mem
        self.g_factory = router
        self.f         = scorer
        self.alpha     = alpha
        self.c         = c
        self.d         = d
        self.rng       = rng

        self.root = CMT.Node(None, rng)

        self.leaf_by_mem_key: Dict[MemKey,CMT.Node] = {}

        self.keyslru = CMT.LRU()

        self.rerouting = False
        self.splitting = False
        self.nodes     = [self.root]

    @property
    def params(self) -> Dict[str,Any]:
        return { 'm': self.max_mem, 'd': self.d, 'c': self.c, "a": self.alpha, "scr": str(self.f), "rou": str(self.g_factory) }

    def __path(self, x, v: 'CMT.Node'):
        path = []
        
        while not v.is_leaf:
            path.append(v)
            v = v.right if v.g.predict(x) > 0 else v.left            

        return CMT.Path(path, v)

    def query(self, x: MemKey, k:int, epsilon:float) -> Tuple[Optional[Tuple[Node, Node, float]], Sequence[Memory], Node]:

        path = self.__path(x, self.root)

        q = self.rng.uniform(0, 1) 
        i = self.rng.randint(0, len(path.nodes))

        if q >= epsilon:
            return (None, path.leaf.topk(x, k, self.f), path.leaf)
 
        elif i < len(path.nodes):
            a = self.rng.choice([path.nodes[i].left, path.nodes[i].right])
            l = self.__path(x, a).leaf
            return ((path.nodes[i], a, 1/2), l.topk(x, k, self.f), l) #update will update the router at path.nodes[i]

        else:
            return ((path.leaf, None, None), path.leaf.randk(k), path.leaf) # update will update the scorer

    def update(self, u: Optional[Tuple[Node, Node, float]], x: MemKey, Z: Sequence[Memory], r: float):

        assert 0 <= r <= 1
        
        if u is None: return

        (v, a, p) = u

        if p is None:
            self.f.update(x, [z[0] for z in Z], r)

            if len(Z) > 0:
                if self.rng.uniform(0, 1) <= r:
                    self.keyslru.remove(Z[0][0])
                    self.keyslru.add(Z[0][0])
        else:
            rhat = (r/p) * (1 if a == v.right else -1)
            B    = log(1e-2 + v.left.n) - log(1e-2 + v.right.n)
            y    = (1 - self.alpha) * rhat + self.alpha * B

            sig_y = 1 if y > 0 else -1
            abs_y = abs(y)

            v.g.update(x, sig_y, abs_y)

        for _ in range(self.d):
            self.__reroute()

    def delete(self, x: MemKey):
        assert x in self.leaf_by_mem_key # deleting something not in the memory ...

        if not self.rerouting:
            self.keyslru.remove(x)

        v = self.leaf_by_mem_key.pop(x)

        while v is not None:
            v.n -= 1
            if v.is_leaf:
                omega = v.memories.pop(x)
            else:
                if v.n == 0:
                    o = v.parent.left if v is v.parent.right else v.parent.right

                    if o.is_leaf:
                        for xprime in o.memories.keys():
                            self.leaf_by_mem_key[xprime] = v.parent

                    v.parent.replace_node(o)

                    self.nodes.pop(self.nodes.index(v))
                    self.nodes.pop(self.nodes.index(o))

                    v = v.parent

            assert v.n >= 0
            v = v.parent

    def insert(self, x: MemKey, omega: MemVal, v: 'CMT.Node' =None):

        assert x not in self.leaf_by_mem_key

        v = v or self.root

        while not v.is_leaf:
            B     = log(1e-2 + v.left.n) - log(1e-2 + v.right.n)
            y     = (1 - self.alpha) * v.g.predict(x) + self.alpha * B
            signy = 1 if y > 0 else -1
            
            v.n += 1
            v.g.update(x, signy, 1)
            v = v.right if v.g.predict(x) > 0 else v.left

        self.__insertLeaf(x, omega, v)

        if not self.rerouting and not self.splitting:
            if self.max_mem is not None and len(self.keyslru) > self.max_mem:
                leastuseful = self.keyslru.peek()
                self.delete(leastuseful)

            for _ in range(self.d):
                self.__reroute()
        
        return v

    def update_omega(self, x: MemKey, newomega:MemVal):
        v = self.leaf_by_mem_key[x]
        assert v.is_leaf
        v.memories[x] = newomega

    def __insertLeaf(self, x: MemKey, omega: MemVal, v: 'CMT.Node'):
        assert v.is_leaf

        if not self.rerouting and not self.splitting:
            self.keyslru.add(x)

        if v.n <= self.c*log(self.root.n+1):
            
            assert x not in self.leaf_by_mem_key
            assert x not in v.memories

            self.leaf_by_mem_key[x] = v
            v.memories[x] = omega
            v.n += 1
            assert v.n == len(v.memories)

        else:            
            self.splitting = True
            mem = v.make_internal(g=self.g_factory())

            self.nodes.append(v.left)
            self.nodes.append(v.right)

            while mem:
                xprime, omegaprime = mem.popitem()
                del self.leaf_by_mem_key[xprime]
                self.insert(xprime, omegaprime, v)

            self.insert(x, omega, v)
            self.splitting = False

        #this is very important for residual learner, less so for memorized learner
        # if not self.rerouting and not self.splitting:
        #     daleaf = self.leaf_by_mem_key[x]
        #     dabest = daleaf.topk(x, 2, self.f)
        #     if len(dabest) > 1:
        #         other = dabest[1] if dabest[0][0] == x else dabest[0]
        #         z = [(x, omega), other]
        #         self.f.update(x, z, 1)

    def __reroute(self):
        x = self.rng.choice(list(self.leaf_by_mem_key.keys()))
        o = self.leaf_by_mem_key[x].memories[x]
        
        self.rerouting = True
        self.delete(x)
        self.insert(x, o)
        self.rerouting = False

    def print(self):

        def print_node(node: CMT.Node, depth:int):
            
            if node is None:
                return

            print( "  "*depth + f"N={len(node.memories)}")
            print_node(node.left, depth+1)
            print_node(node.right, depth+1)

        print_node(self.root, 0)

    def walk(self, func):

        def walk_node(node: CMT.Node):
            
            if node is None:
                return

            func(node)

            walk_node(node.left)
            walk_node(node.right)

        walk_node(self.root)
