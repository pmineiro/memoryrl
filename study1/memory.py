from math      import log
from itertools import count
from typing    import Dict, Any, Sequence, Tuple, Hashable, Iterable, List

from routers   import RouterFactory, Router
from scorers   import Scorer
from splitters import Splitter

from coba.random import CobaRandom

MemKey = Hashable
MemVal = Any
Memory = Tuple[MemKey,MemVal]

class Node:
    def __init__(self, id: int, parent: 'Node'):
        self.parent = parent

        self.memories: Dict[MemKey,MemVal] = {}

        self.id            = id
        self.n             = 0
        self.left : Node   = None
        self.right: Node   = None
        self.g    : Router = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    @property
    def depth(self):
        return 1 + self.parent.depth if self.parent else 0

class CMT:

    def __init__(self,
        max_mem:int,
        router: RouterFactory,
        scorer: Scorer,
        c:Splitter,
        d:int,
        alpha:float=0.25, 
        rng:CobaRandom= CobaRandom(1337)):

        self.max_mem      = max_mem
        self.g_factory = router
        self.f         = scorer
        self.alpha     = alpha
        self.c         = c
        self.d         = d
        self.rng       = rng
        self.node_ids  = iter(count())

        self.root = Node(next(self.node_ids), None)

        self.leaf_by_key: Dict[MemKey,Node] = {}
        self.nodes: List[Node] = [self.root]

        self.rerouting = False
        self.splitting = False

    @property
    def times(self) -> Sequence[float]:
        return self.f.times

    @property
    def params(self) -> Dict[str,Any]:
        return { 'm': self.max_mem, 'd': self.d, 'c': str(self.c), "a": self.alpha, "scr": str(self.f), "rou": str(self.g_factory) }

    def query(self, key) -> MemVal:
        
        if self.root.n == 0: return 0
        
        return self.__query(key, self.root)[1]

    def update(self, key: MemKey, outcome: float, weight: float) -> None:

        assert 0 <= outcome <= 1
        assert 0 <= weight

        if self.root.n == 0: return

        query_path, query_pred = self.__query(key, self.root)

        #update leaf
        mem_key_err_pairs = [ (k, (outcome-v)**2) for k,v in query_path[-1].memories.items() ]
        mem_keys, mem_errs = zip(*mem_key_err_pairs)
        self.f.update(key, mem_keys, mem_errs, weight)

        #update routers
        for curr_node, next_node in zip(query_path, query_path[1:]):

            alternate_node = curr_node.left if curr_node.right is next_node else curr_node.right
            alternate_pred = self.__query(key,alternate_node)[1] 

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
        
        v.n -=1
        v.memories.pop(key)
        v = v.parent

        while v is not None:
            v.n -= 1

            child_sans_memory = v.left if v.left.n == 0 else v.right if v.right.n == 0 else None

            if child_sans_memory is not None:

                child_with_memory = v.right if child_sans_memory is v.left else v.left

                assert child_with_memory.n != 0

                child_with_memory.parent = v.parent

                if v.parent and v.parent.right is v:
                    v.parent.right = child_with_memory

                if v.parent and v.parent.left is v:
                    v.parent.left = child_with_memory

                if v.parent is None:
                    self.root = child_with_memory

                self.nodes.pop(self.nodes.index(v))
                self.nodes.pop(self.nodes.index(child_sans_memory))

            assert v.n >= 0
            v = v.parent

    def insert(self, key: MemKey, value: MemVal, weight: float, *, v: Node=None):

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

    def __insertLeaf(self, key: MemKey, val: MemVal, leaf: Node, weight: float):

        assert leaf.is_leaf

        if leaf.n <= self.c(self.root.n):

            assert key not in self.leaf_by_key
            assert key not in leaf.memories

            self.leaf_by_key[key] = leaf
            leaf.memories[key] = val
            leaf.n += 1
            assert leaf.n == len(leaf.memories)

        else:
            self.splitting   = True

            to_insert = list(leaf.memories.items())
            leaf.memories.clear()

            new_parent       = leaf
            new_parent.left  = Node(next(self.node_ids), new_parent)
            new_parent.right = Node(next(self.node_ids), new_parent)
            new_parent.n     = 0
            new_parent.g     = self.g_factory()

            self.nodes.append(new_parent.left)
            self.nodes.append(new_parent.right)

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

    def __path(self, key: MemKey, node: Node) -> Iterable[Node]:
        yield node
        while not node.is_leaf:
            node = node.right if node.g.predict(key) > 0 else node.left
            yield node

    def __query(self, key: MemKey, init: Node) -> Tuple[List[Node], MemVal]:
        query_path = list(self.__path(key, init))
        final_node = query_path[-1]

        assert final_node.is_leaf and final_node.memories

        mem_keys    = list(final_node.memories.keys())
        mem_scores  = self.f.predict(key,mem_keys)
        tie_breaker = ( self.rng.random() for _ in count()) #randomly break

        top_mem_key = list(zip(*sorted(zip(mem_scores, tie_breaker, mem_keys))))[2][0]
        query_pred  = final_node.memories[top_mem_key]

        return query_path, query_pred

    def __update_g(self, key: MemKey,  left_err: float, right_err: float, v: Node, weight: float):

        assert 0 <= left_err and left_err <=1
        assert 0 <= right_err and right_err <=1

        balance_diff = log(1e-2 + v.left.n) - log(1e-2 + v.right.n)
        error_diff   = left_err-right_err

        label = 1 if (1-self.alpha) * error_diff + self.alpha * balance_diff > 0 else -1
        
        if error_diff != 0:
            v.g.update(key, label, weight*abs(error_diff))
