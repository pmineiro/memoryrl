from math      import log
from itertools import count
from typing    import Dict, Any, Tuple, Hashable, Iterable, List

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
        self.left : Node   = None
        self.right: Node   = None
        self.g    : Router = None
        self.n    : int    = 0
    
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
        v:int = 1,
        rng:CobaRandom= CobaRandom(1337)):

        self.max_mem   = max_mem
        self.g_factory = router
        self.f         = scorer
        self.alpha     = alpha
        self.c         = c
        self.d         = d
        self.rng       = rng
        self.node_ids  = iter(count())
        self.v         = v

        self.root = Node(next(self.node_ids), None)

        self.leaf_by_key: Dict[MemKey,Node] = {}
        self.nodes: List[Node] = [self.root]

        self.rerouting = False
        self.splitting = False

    @property
    def params(self) -> Dict[str,Any]:
        return { 'v':self.v, 'm': self.max_mem, 'd': self.d, 'c': str(self.c), "a": self.alpha, "scr": str(self.f), "rou": str(self.g_factory) }

    def query(self, key) -> MemVal:

        if self.root.n == 0: return 0

        return self.__query(key, self.root)[1]

    def insert(self, key: MemKey, value: MemVal, weight: float, *, node: Node=None):

        if key in self.leaf_by_key: return

        node = node or self.root

        leaf = self.__update_routers(node, key, value, weight, mode="insert")
        leaf = self.__insert_leaf(leaf, key, value, weight)

        if not self.splitting:
            self.__update_scorer(leaf, key, value, weight, mode="insert")

        if not self.rerouting and not self.splitting:
            self.__reroute()

    def update(self, key: MemKey, outcome: float, weight: float) -> None:
        assert 0 <= outcome <= 1
        assert 0 <= weight

        if self.root.n == 0: return

        leaf = self.__update_routers(self.root, key, outcome, weight, mode="update")
        self.__update_scorer(leaf, key, outcome, weight, mode="update")
        self.__reroute()

    def delete(self, key: MemKey):
        assert key in self.leaf_by_key # deleting something not in the memory ...

        v = self.leaf_by_key.pop(key)
        v.memories.pop(key)

        v.n -= 1
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

    def __insert_leaf(self, leaf: Node, insert_key: MemKey, insert_val: MemVal, weight: float) -> Node:

        assert leaf.is_leaf

        if leaf.n <= self.c(self.root.n):

            assert insert_key not in self.leaf_by_key
            assert insert_key not in leaf.memories

            self.leaf_by_key[insert_key] = leaf
            leaf.memories[insert_key] = insert_val

            leaf.n += 1

            if not self.splitting:
                parent = leaf.parent
                while parent is not None:
                    parent.n += 1
                    parent = parent.parent

            assert leaf.n == len(leaf.memories)

        else:
            self.splitting   = True
            print(f"SPLITTING {self.root.n} {leaf.id}")
            new_parent       = leaf
            new_parent.left  = Node(next(self.node_ids), new_parent)
            new_parent.right = Node(next(self.node_ids), new_parent)
            new_parent.g     = self.g_factory()

            self.nodes.append(new_parent.left)
            self.nodes.append(new_parent.right)

            split_keys = list(leaf.memories.keys())
            split_vals = list(leaf.memories.values())
            
            for _ in range(1): # it is possible by adjusting this to re-split the same memories multiple times
                
                #split_keys,split_vals = tuple(zip(*self.rng.shuffle(list(zip(split_keys,split_vals)))))

                for split_key in split_keys:
                    split_leaf = self.leaf_by_key.pop(split_key)
                    split_leaf.n-=1
                    split_leaf.memories.pop(split_key)

                for split_key, split_val in zip(split_keys,split_vals):
                    self.insert(split_key, split_val, 1, node=new_parent)
            
            leaf.n += len(split_keys)
            self.splitting = False
            self.insert(insert_key, insert_val, weight, node=new_parent)
        
        return self.leaf_by_key[insert_key]

    def __reroute(self):

        _d = self.d if self.d >= 1 else 1 if self.rng.random() < self.d else 0

        for _ in range(_d):

            x = self.rng.choice(list(self.leaf_by_key.keys()))
            o = self.leaf_by_key[x].memories[x]

            self.rerouting = True
            old_n = self.root.n
            self.delete(x)
            assert self.root.n == old_n-1
            self.insert(x, o, 1)
            assert self.root.n == old_n
            self.rerouting = False

    def __path(self, key: MemKey, node: Node) -> Iterable[Node]:
        yield node
        while not node.is_leaf:
            label = node.g.predict(key) or self.rng.choice([-1,1])
            node  = node.left if label < 0 else node.right 
            yield node

    def __query(self, key: MemKey, init: Node) -> Tuple[MemKey, MemVal, float]:
        final_node = list(self.__path(key, init))[-1]

        assert (final_node.is_leaf and final_node.memories) or self.splitting

        memories = final_node.memories

        if not memories:
            top_mem_key   = None
            top_mem_val   = None
            top_mem_score = None
        else:
            mem_keys     = list(memories.keys())
            mem_scores   = self.f.predict(key, mem_keys)
            tie_breakers = self.rng.randoms(len(mem_scores)) #randomly break
            sorted_mems  = list(sorted(zip(mem_scores, tie_breakers, mem_keys)))

            top_mem_key   = sorted_mems[0][2]
            top_mem_val   = memories[top_mem_key]
            top_mem_score = sorted_mems[0][0]            

        return top_mem_key, top_mem_val, top_mem_score

    def __update_router(self, node: Node, key: MemKey,  left_err: float, right_err: float, weight: float):

        assert 0 <= left_err and left_err <=1
        assert 0 <= right_err and right_err <=1

        balance_direction = log(1e-2 + node.left.n) - log(1e-2 + node.right.n)
        error_direction   = left_err-right_err
        final_direction  = (1-self.alpha) * error_direction + self.alpha * balance_direction 

        if final_direction != 0:
            label = 1 if final_direction > 0 else -1
            node.g.update(key, label, weight*abs(final_direction))

    def __update_routers(self, node:Node, key: MemKey, val: MemVal, weight:float, mode:str) -> Node:

        assert mode in ['update','insert']

        while not node.is_leaf:

            _, left_val, left_score = self.__query(key, node.left)
            _, right_val, right_score = self.__query(key, node.right)

            if self.v == 1 :
                left_loss  = (val-left_val)**2 if left_val is not None else 0
                right_loss = (val-right_val)**2 if right_val is not None else 0

            if self.v == 2:
                left_loss  = left_score if left_score is not None else 0
                right_loss = right_score if right_score is not None else 0

            if mode == "update": 
                direction = node.g.predict(key)

            #this if statement seems to improve our performance a 
            if self.splitting or self.rerouting or mode == "update":
                self.__update_router(node, key, left_loss, right_loss, weight)

            if mode == "insert": 
                direction = node.g.predict(key)

            direction = direction or self.rng.choice([-1,1])
            node = node.left if direction < 0 else node.right

        return node

    def __update_scorer(self, leaf: Node, key: MemKey, val: MemVal, weight:float, mode:str):

        assert leaf.is_leaf
        assert mode in ['update','insert'] 

        mem_key_err_pairs = [ (k, (val-v)**2) for k,v in leaf.memories.items() ]
        mem_keys, mem_errs = zip(*mem_key_err_pairs)
        self.f.update(key, mem_keys, mem_errs, weight)
