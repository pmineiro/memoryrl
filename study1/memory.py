from math   import log
from heapq  import heappush, heappop
from typing import Dict, Any, Tuple, Hashable, Iterable, List

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

        def top(self, x, f:Scorer) -> MemVal:
            assert self.is_leaf
            #this could be modified so that it randomly picks in the case of ties. 

            if not self.memories:
                return None

            keys   = list(self.memories.keys())
            scores = f.predict(x, keys)

            return self.memories[min(zip(keys,scores), key=lambda t: t[1])[0]]

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
        self.leaf_by_key: Dict[MemKey,CMT.Node] = {}
        self.nodes = [self.root]

        self.rerouting = False
        self.splitting = False

    @property
    def params(self) -> Dict[str,Any]:
        return { 'm': self.max_mem, 'd': self.d, 'c': self.c, "a": self.alpha, "scr": str(self.f), "rou": str(self.g_factory) }

    def query(self, key) -> MemVal:
        return self.__query(key) 

    def update(self, key: MemKey, outcome: float) -> None:

        assert 0 <= outcome <= 1

        query_path, query_pred = self.__query(key)

        if query_pred is None: return

        for i in range(len(query_path)):

            if query_path[i].is_leaf:                
                key_and_pred_err = [ (k, (outcome-v)**2) for k,v in query_path[i].memories.items() ]
                self.f.update(key, *zip(*key_and_pred_err))

            else:

                this_node = query_path[i  ]
                next_node = query_path[i+1]

                alternate_node = this_node.left if this_node.right is next_node else this_node.right
                alternate_pred = list(self.__path(key,alternate_node))[-1].top(key, self.f)

                left_pred  = query_pred if this_node.left  is next_node else alternate_pred
                right_pred = query_pred if this_node.right is next_node else alternate_pred

                #???? When one side is empty do I learn nothing or strengthen the non-empty side
                left_pred_err  = (outcome-left_pred)**2 if left_pred is not None else 1
                right_pred_err = (outcome-right_pred)**2 if right_pred is not None else 1

                self.__update_g(key, left_pred_err, right_pred_err, this_node)

        for _ in range(self.d):
            self.__reroute()

    def delete(self, key: MemKey):
        assert key in self.leaf_by_key # deleting something not in the memory ...

        v = self.leaf_by_key.pop(key)

        while v is not None:
            v.n -= 1
            if v.is_leaf:
                omega = v.memories.pop(key)
            else:
                if v.n == 0:
                    o = v.parent.left if v is v.parent.right else v.parent.right

                    if o.is_leaf:
                        for xprime in o.memories.keys():
                            self.leaf_by_key[xprime] = v.parent

                    v.parent.replace_node(o)

                    self.nodes.pop(self.nodes.index(v))
                    self.nodes.pop(self.nodes.index(o))

                    v = v.parent

            assert v.n >= 0
            v = v.parent

    def insert(self, key: MemKey, value: MemVal, v: 'CMT.Node'=None):

        if key in self.leaf_by_key: return

        v = v or self.root

        while not v.is_leaf:

            v.n += 1

            left_error  = 1 if v.g.predict(key) > 0 else 0
            right_error = 1-left_error
            self.__update_g(key, left_error, right_error, v)

            v = v.right if v.g.predict(key) > 0 else v.left

        self.__insertLeaf(key, value, v)

        if not self.rerouting and not self.splitting:
            for _ in range(self.d):
                self.__reroute()

    def __insertLeaf(self, key: MemKey, omega: MemVal, leaf: 'CMT.Node'):

        assert leaf.is_leaf

        #if leaf.n <= self.c*log(self.root.n+1):
        if leaf.n <= self.c:

            assert key not in self.leaf_by_key
            assert key not in leaf.memories

            self.leaf_by_key[key] = leaf
            leaf.memories[key] = omega
            leaf.n += 1
            assert leaf.n == len(leaf.memories)

        else:
            print("SPLITTING")
            self.splitting = True
            mem = leaf.make_internal(g=self.g_factory())

            self.nodes.append(leaf.left)
            self.nodes.append(leaf.right)

            while mem:
                xprime, omegaprime = mem.popitem()
                del self.leaf_by_key[xprime]
                self.insert(xprime, omegaprime, leaf)

            self.insert(key, omega, leaf)
            self.splitting = False

        mem_keys = self.leaf_by_key[key].memories.keys()
        mem_errs = [ 0 if k == key else 1 for k in mem_keys]
        self.f.update(key, mem_keys, mem_errs)

    def __reroute(self):
        if self.leaf_by_key:
            x = self.rng.choice(list(self.leaf_by_key.keys()))
            o = self.leaf_by_key[x].memories[x]
            
            self.rerouting = True
            self.delete(x)
            self.insert(x, o)
            self.rerouting = False

    def __path(self, key: MemKey, node: 'CMT.Node') -> Iterable['CMT.Node']:
        yield node
        while not node.is_leaf:
            node = node.right if node.g.predict(key) > 0 else node.left
            yield node

    def __query(self, key: MemKey) -> Tuple[List['CMT.Node'], MemVal]:
        query_path = list(self.__path(key, self.root))
        query_pred = query_path[-1].top(key, self.f)        
        return query_path, query_pred

    def __update_g(self, key: MemKey,  left_err: float, right_err: float, v: 'CMT.Node'):

        assert 0 <= left_err and left_err <=1
        assert 0 <= right_err and right_err <=1

        balance_diff = log(1e-2 + v.left.n) - log(1e-2 + v.right.n)
        
        error_diff   = left_err-right_err

        label = 1 if (1-self.alpha) * error_diff + self.alpha * balance_diff > 0 else -1
        v.g.update(key, label, abs(error_diff))
