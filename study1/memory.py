from math   import log
from typing import Dict, Any, Tuple, Hashable, Iterable, List

from routers import RouterFactory
from scorers import Scorer
from random  import Random
from splitters import Splitter

from coba.random import CobaRandom

MemKey = Hashable
MemVal = Any
Memory = Tuple[MemKey,MemVal]

class CMT:

    class Node:
        def __init__(self, parent:'CMT.Node', rng: Random):
            self.parent = parent

            self.memories: Dict[MemKey,MemVal] = {}

            self.rng = rng

            self.n     = 0
            self.left  = None
            self.right = None
            self.g     = None

        @property
        def is_leaf(self) -> bool:
            return self.left is None

        @property
        def depth(self):
            return 1 + self.parent.depth if self.parent else 0

        def top(self, x, f:Scorer) -> MemVal:
            assert self.is_leaf
            #this could be modified so that it randomly picks in the case of ties. 

            if not self.memories:
                return None

            keys   = list(self.memories.keys())
            scores = f.predict(x, keys)
            sort   = sorted(zip(keys,scores), key=lambda t: (t[1], self.rng.random()))

            return self.memories[sort[0][0]]

    def __init__(self, 
        max_mem:int, 
        router: RouterFactory, 
        scorer: Scorer, 
        c:Splitter, 
        d:int, 
        alpha:float=0.25, 
        rng:CobaRandom= CobaRandom(1337)):

        self.max_mem   = max_mem
        self.g_factory = router
        self.f         = scorer
        self.alpha     = alpha
        self.c         = c
        self.d         = d
        self.rng       = rng

        self.root = CMT.Node(None, rng)

        self.leaf_by_key: Dict[MemKey,CMT.Node] = {}
        self.nodes: List[CMT.Node] = [self.root]

        self.rerouting = False
        self.splitting = False

    @property
    def params(self) -> Dict[str,Any]:
        return { 'm': self.max_mem, 'd': self.d, 'c': str(self.c), "a": self.alpha, "scr": str(self.f), "rou": str(self.g_factory) }

    def query(self, key) -> MemVal:
        return self.__query(key) 

    def update(self, key: MemKey, outcome: float, weight: float) -> None:

        assert 0 <= outcome <= 1
        assert 0 <= weight

        query_path, query_pred = self.__query(key)

        if query_pred is None: return

        #update leaf
        mem_key_err_pairs = [ (k, (outcome-v)**2) for k,v in query_path[-1].memories.items() ]
        mem_keys, mem_errs = zip(*mem_key_err_pairs)
        self.f.update(key, mem_keys, mem_errs, weight)

        #update routers
        for curr_node, next_node in zip(query_path, query_path[1:]):

            alternate_node = curr_node.left if curr_node.right is next_node else curr_node.right
            alternate_pred = list(self.__path(key,alternate_node))[-1].top(key, self.f)

            assert (curr_node.left is next_node) or (curr_node.right is next_node)
            assert (curr_node.left is alternate_node) or (curr_node.right is alternate_node)
            assert alternate_node is not next_node

            left_pred  = query_pred if curr_node.left  is next_node else alternate_pred
            right_pred = query_pred if curr_node.right is next_node else alternate_pred

            left_pred_err  = (outcome-left_pred)**2
            right_pred_err = (outcome-right_pred)**2

            self.__update_g(key, left_pred_err, right_pred_err, curr_node, weight)

        self.__reroute()

    def delete(self, key: MemKey):
        assert key in self.leaf_by_key # deleting something not in the memory ...

        v = self.leaf_by_key.pop(key)

        while v is not None:                
            v.n -= 1
            
            if v.is_leaf:
                v.memories.pop(key)
            
            else:
                null_child = v.left if v.left.n == 0 else v.right if v.right.n == 0 else None
                live_child = v.right if v.left.n == 0 else v.left if v.right.n == 0 else None

                if null_child is not None:

                    live_child.parent = v.parent

                    if v.parent and v.parent.right is v:
                        v.parent.right = live_child
                    
                    if v.parent and v.parent.left is v:
                        v.parent.left = live_child

                    if v.parent is None:
                        self.root = live_child

                    self.nodes.pop(self.nodes.index(v))
                    self.nodes.pop(self.nodes.index(null_child))

                    v = v.parent

            assert v.n >= 0
            v = v.parent

    def insert(self, key: MemKey, value: MemVal, weight: float, *, v: 'CMT.Node'=None):

        if key in self.leaf_by_key: return

        v = v or self.root

        while not v.is_leaf:

            v.n += 1

            left_error  = 1 if v.g.predict(key) > 0 else 0
            right_error = 1-left_error
            self.__update_g(key, left_error, right_error, v, weight)

            v = v.right if v.g.predict(key) > 0 else v.left

        self.__insertLeaf(key, value, v, weight)

        if not self.rerouting and not self.splitting:
            self.__reroute()

    def __insertLeaf(self, key: MemKey, val: MemVal, leaf: 'CMT.Node', weight: float):

        assert leaf.is_leaf

        if leaf.n <= self.c(self.root.n):

            assert key not in self.leaf_by_key
            assert key not in leaf.memories

            self.leaf_by_key[key] = leaf
            leaf.memories[key] = val
            leaf.n += 1
            assert leaf.n == len(leaf.memories)

        else:
            print("SPLITTING")
            self.splitting = True

            new_parent       = leaf
            new_parent.left  = CMT.Node(new_parent, self.rng)
            new_parent.right = CMT.Node(new_parent, self.rng)
            new_parent.n     = 0
            new_parent.g     = self.g_factory()

            self.nodes.append(new_parent.left)
            self.nodes.append(new_parent.right)

            to_insert = list(new_parent.memories.items())
            new_parent.memories.clear()

            for mem_key, mem_val in to_insert:
                self.leaf_by_key.pop(mem_key,None)
                self.insert(mem_key, mem_val, 1, v=new_parent)

            self.insert(key, val, weight, v=new_parent)
            self.splitting = False

        if not self.splitting:
            mem_keys = self.leaf_by_key[key].memories.keys()
            mem_errs = [ 0 if k == key else 1 for k in mem_keys]
            self.f.update(key, mem_keys, mem_errs, weight)

    def __reroute(self):

        _d = self.d if self.d >= 1 else 1 if self.rng.random() < self.d else 0

        for _ in range(_d):
            x = self.rng.choice(list(self.leaf_by_key.keys()))
            o = self.leaf_by_key[x].memories[x]

            self.rerouting = True
            self.delete(x)
            self.insert(x, o, 1)
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

    def __update_g(self, key: MemKey,  left_err: float, right_err: float, v: 'CMT.Node', weight: float):

        assert 0 <= left_err and left_err <=1
        assert 0 <= right_err and right_err <=1

        balance_diff = log(1e-2 + v.left.n) - log(1e-2 + v.right.n)
        error_diff   = left_err-right_err

        label = 1 if (1-self.alpha) * error_diff + self.alpha * balance_diff > 0 else -1
        
        if error_diff != 0:
            v.g.update(key, label, weight*abs(error_diff))
