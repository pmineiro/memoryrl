from math        import log
from itertools   import count
from collections import Counter
from typing      import Dict, Any, Tuple, Hashable, Iterable

from routers   import RouterFactory, Router, ProjRouter
from scorers   import Scorer, RandomScorer
from splitters import Splitter

from coba.random import CobaRandom

MemKey   = Hashable
MemVal   = Any
MemScore = float
Memory   = Tuple[MemKey,MemVal]

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
        v:Tuple[int,...] = (1,),
        mlr:float = 0,
        rng:int = 1337):

        self.max_mem   = max_mem
        self.g_factory = router
        self.f         = scorer
        self.alpha     = alpha
        self.c         = c
        self.d         = d
        self.rng       = CobaRandom(rng)
        self.node_ids  = iter(count())
        self.v         = v
        self.mlr       = mlr

        self.root = Node(next(self.node_ids), None)
        self.leaf_by_key: Dict[MemKey,Node] = {}

        self.rerouting = False
        self.splitting = False

    @property
    def params(self) -> Dict[str,Any]:
        return { 'type':'CMT', 'v':self.v, 'm': self.max_mem, 'c': str(self.c), 'd': self.d, "a": self.alpha, "scr": str(self.f), "rou": str(self.g_factory) }

    def query(self, key) -> Tuple[MemVal,MemScore]:

        if self.root.n == 0: return [0,0]

        return self.__query(key, self.root)[1:3]

    def insert(self, key: MemKey, value: MemVal, weight: float):

        if key in self.leaf_by_key: return

        self.__update_routers(key, value, weight, mode="insert")
        self.__insert_leaf(list(self._path(key,self.root))[-1], key, value, weight)
        self.__update_scorer(self.leaf_by_key[key], key, value, weight, mode="insert")
        #self.__reroute()

    def update(self, key: MemKey, outcome: float, weight: float) -> None:
        assert 0 <= outcome <= 1
        assert 0 <= weight

        if self.root.n == 0: return

        self.__update_omega(key, outcome)
        self.__update_routers(key, outcome, weight, mode="update")
        self.__update_scorer(list(self._path(key,self.root))[-1], key, outcome, weight, mode="update")
        self.__reroute()

    def delete(self, key: MemKey) -> None:
        assert key in self.leaf_by_key # deleting something not in the memory ...

        leaf = self.leaf_by_key.pop(key)
        leaf.memories.pop(key)

        node = leaf
        while node is not None:
            node.n -= 1
            node = node.parent

        if leaf.n == 0 and leaf.parent:
            prune_child  = leaf
            other_child  = leaf.parent.left if leaf.parent.right is leaf else leaf.parent.right
            child_parent = leaf.parent
            grand_parent = prune_child.parent.parent

            if grand_parent and grand_parent.left is child_parent:
                grand_parent.left = other_child
                other_child.parent = grand_parent

            if grand_parent and grand_parent.right is child_parent:
                grand_parent.right = other_child
                other_child.parent = grand_parent

            if not grand_parent:
                self.root = other_child
                other_child.parent = None

    def __update_omega(self, key: MemKey, outcome:float) -> None:
        if self.mlr > 0:
            top_key, top_val = self.__query(key,self.root)[0:2]
            self.leaf_by_key[top_key].memories[top_key] = self.mlr*outcome + (1-self.mlr)*top_val

    def __insert_leaf(self, leaf: Node, insert_key: MemKey, insert_val: MemVal, weight:float) -> None:

        assert leaf.is_leaf

        if leaf.n <= self.c(self.root.n):

            assert insert_key not in self.leaf_by_key
            assert insert_key not in leaf.memories

            self.leaf_by_key[insert_key] = leaf
            leaf.memories[insert_key] = insert_val

            node = leaf
            while node is not None:
                node.n += 1
                node = node.parent

            assert leaf.n == len(leaf.memories)

        else:
            split_memories = leaf.memories
            split_keys     = list(split_memories.keys())

            new_parent          = leaf
            new_parent.left     = Node(next(self.node_ids), new_parent)
            new_parent.right    = Node(next(self.node_ids), new_parent)
            new_parent.g        = self.g_factory.create(split_keys)
            new_parent.memories = dict()

            for split_key in split_keys:
                leaf = new_parent.left if new_parent.g.predict(split_key) < 0 else new_parent.right
                leaf.n+=1
                leaf.memories[split_key] = split_memories[split_key]
                self.leaf_by_key[split_key] = leaf

            # for split_key,split_val in new_parent.left.memories.items():
            #     self.__update_scorer(new_parent.left, split_key, split_val, 1, mode="insert")

            # for split_key,split_val in new_parent.right.memories.items():
            #     self.__update_scorer(new_parent.right, split_key, split_val, 1, mode="insert")

            direction = new_parent.g.predict(insert_key) or self.rng.choice([-1,1])
            mode      = "insert"
            self.__update_router(new_parent, insert_key, insert_val, direction, mode, weight)
            leaf = new_parent.left if direction < 0 else new_parent.right
            self.__insert_leaf(leaf, insert_key, insert_val, weight)

    def __reroute(self) -> None:

        if self.rerouting: return
        if len(self.leaf_by_key) < 3: return

        flt_part = self.d - int(self.d)
        int_part = int(self.d)

        for _ in range(int_part + int(self.rng.random() < flt_part)):

            prune_leaf = self.root

            while not prune_leaf.is_leaf:
                prune_leaf = [prune_leaf.left, prune_leaf.right][self.rng.randint(0,1)]

            prune_mems = prune_leaf.memories.copy()

            self.rerouting = True

            for key in prune_mems: self.delete(key)
            for key,val in prune_mems.items(): self.insert(key,val,1)

            self.rerouting = False

    def _path(self, key: MemKey, node: Node) -> Iterable[Node]:
        yield node
        while not node.is_leaf:
            label = node.g.predict(key) or self.rng.choice([-1,1])
            node  = node.left if label < 0 else node.right 
            yield node

    def __query(self, key: MemKey, init: Node) -> Tuple[MemKey, MemVal, MemScore]:
        
        final_node = list(self._path(key, init))[-1]
        assert (final_node.is_leaf and final_node.memories)
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

    def __update_router(self, node: Node, key: MemKey, val: MemVal, direction: float, mode:str, weight: float) -> None:

        assert not node.is_leaf

        if not isinstance(self.g_factory,ProjRouter):
            if self.v[0] == 1:
                _, left_val, left_score = self.__query(key, node.left)
                _, right_val, right_score = self.__query(key, node.right)

                left_loss  = (val-left_val)**2 if left_val is not None else 0
                right_loss = (val-right_val)**2 if right_val is not None else 0

            if self.v[0] == 2:
                _, left_val, left_score = self.__query(key, node.left)
                _, right_val, right_score = self.__query(key, node.right)

                left_loss  = left_score if left_score is not None else 0
                right_loss = right_score if right_score is not None else 0

            assert 0 <= left_loss and left_loss <=1
            assert 0 <= right_loss and right_loss <=1

            balance_direction = log(1e-2 + node.left.n) - log(1e-2 + node.right.n)
            outcome_direction = left_loss-right_loss
            combine_direction = outcome_direction + self.alpha * (balance_direction-outcome_direction) 

            if combine_direction == 0: return

            label = -1 if combine_direction <= 0 else 1
            node.g.update(key, label, weight*abs(combine_direction))

    def __update_routers(self, key: MemKey, val: MemVal, weight:float, mode:str) -> None:

        if isinstance(self.g_factory, ProjRouter): return

        assert mode in ['update','insert']

        node = self.root

        while not node.is_leaf:

            direction = node.g.predict(key) or self.rng.choice([-1,1])

            self.__update_router(node, key, val, direction, mode, weight)

            if mode == "insert": direction = node.g.predict(key) or self.rng.choice([-1,1])

            direction = direction or self.rng.choice([-1,1])
            node      = node.left if direction < 0 else node.right

    def __update_scorer(self, leaf: Node, key: MemKey, val: MemVal, weight:float, mode:str) -> None:

        assert leaf.is_leaf
        assert mode in ['update','insert'] 

        mem_key_err_pairs = [ (k, (val-v)**2) for k,v in leaf.memories.items() ]
        mem_keys, mem_errs = zip(*mem_key_err_pairs)
        
        self.f.update(key, mem_keys, mem_errs, weight)

class CMF:

    def __init__(self,
        n_trees:int,
        max_mem:int,
        router: RouterFactory,
        scorer: Scorer,
        c:Splitter,
        d:int,
        alpha:float=0.25, 
        v:Tuple[int,...] = (1,),
        mlr:float = 0) -> None:

        self._trees  = [ CMT(max_mem, router, RandomScorer(), c, d, alpha, v, mlr, i) for i in range(n_trees)]
        self._scorer = scorer

        self.max_mem   = max_mem
        self.g_factory = router
        self.f         = scorer
        self.alpha     = alpha
        self.c         = c
        self.d         = d
        self.node_ids  = iter(count())
        self.v         = v
        self.mlr       = mlr
        self.rng       = CobaRandom(1337)
        self.root      = None

    @property
    def params(self) -> Dict[str,Any]:
        return { 'T': len(self._trees), 'type':'CMF', 'v':self.v, 'm': self.max_mem, 'c': str(self.c), 'd': self.d, "a": self.alpha, "scr": str(self.f), "rou": str(self.g_factory) }

    def query(self, key: MemKey) -> Tuple[MemVal,MemScore]:

        memories = self.__memories(key)

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

        return top_mem_val, top_mem_score

    def insert(self, key: MemKey, value: MemVal, weight: float):

        for tree in self._trees:
            tree.insert(key,value,weight)

        self.__update_scorer(key, value, weight, "insert")

    def update(self, key: MemKey, outcome: float, weight: float) -> None:

        for tree in self._trees:
            tree.update(key,outcome,weight)

        self.__update_scorer(key, outcome, weight, "update")

    def __memories(self, key: MemKey) -> Dict[MemKey,MemVal]:
        if self._trees[0].root.n == 0: 
            return {}

        memories = {}
        votes = Counter()

        for tree in self._trees:
            tree_memories = list(tree._path(key, tree.root))[-1].memories
            memories.update(tree_memories)
            votes = votes + Counter(tree_memories.keys())

        return { k: memories[k] for k,v in  sorted(votes.items(), key=lambda k: k[1], reverse=True)[:75] }

    def __update_scorer(self, key: MemKey, val: MemVal, weight:float, mode:str) -> None:

        assert mode in ['update','insert'] 

        mem_key_err_pairs = [ (k, (val-v)**2) for k,v in self.__memories(key).items() ]
        
        if len(mem_key_err_pairs) == 0: return

        mem_keys, mem_errs = zip(*mem_key_err_pairs)
        self.f.update(key, mem_keys, mem_errs, weight)