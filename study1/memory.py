from math        import log
from itertools   import count
from typing      import Dict, Any, Tuple, Hashable, Iterable, Sequence

from routers   import RouterFactory, Router, EigenRouter, VowpRouter
from scorers   import Scorer
from splitters import Splitter
from bounders  import Bounder

from coba.random import CobaRandom
from coba.learners import VowpalMediator

MemKey   = Hashable
MemVal   = Any
MemScore = float
Memory   = Tuple[MemKey,MemVal]


class EMT_PY:
    """The Python implementation of EMT exists to test new ideas more easily. """
    class Node:
        def __init__(self, id: int, parent: 'EMT_PY.Node'):
            self.parent = parent

            self.memories: Dict[MemKey,MemVal] = {}

            self.id                 = id
            self.left : EMT_PY.Node = None
            self.right: EMT_PY.Node = None
            self.g    : Router      = None
            self.n    : int         = 0
        
        @property
        def is_leaf(self) -> bool:
            return self.left is None

        @property
        def depth(self):
            return 1 + self.parent.depth if self.parent else 1

    def __init__(self,
        bounder       : Bounder,
        router_factory: RouterFactory,
        scorer        : Scorer,
        splitter      : Splitter,
        d             : int,
        alpha         : float,
        rng           : int = 1337) -> None:

        self.bounder        = bounder
        self.router_factory = router_factory
        self.scorer         = scorer
        self.splitter       = splitter
        self.d              = d
        self.alpha          = alpha
        self.rng            = CobaRandom(rng)
        self.node_ids       = iter(count())

        self.root = EMT_PY.Node(next(self.node_ids), None)
        self.leaf_by_key: Dict[MemKey,EMT_PY.Node] = {}

        self.rerouting = False
        self.splitting = False

    @property
    def params(self) -> Dict[str,Any]:
        return { 'type':'EMT_PY', 'b': str(self.bounder), 'c': str(self.splitter), 'f': str(self.scorer), 'g': str(self.router_factory), 'd': self.d, 'a': self.alpha }

    def predict(self, key) -> Tuple[MemVal,MemScore]:
        
        if self.root.n == 0: return [0,0]
        _key,_val,_score = self.__query(key, self.root)
        
        self.__bounded(_key)
        return _val,_score

    def learn(self, key: MemKey, value: MemVal, weight: float):

        if key in self.leaf_by_key: return

        self.__update_scorer(list(self.__path(key,self.root))[-1], key, value, weight, mode="update")
        self.__update_routers(key, value, weight, mode="insert")
        self.__insert_leaf(list(self.__path(key,self.root))[-1], key, value, weight)

        #I should test with and without this update...
        #self.__update_scorer(self.leaf_by_key[key], key, value, weight, mode="insert")

        self.__bounded(key)
        self.__reroute()

    def memories(self, key: MemKey) -> Dict[MemKey, MemVal]:
        return list(self.__path(key, self.root))[-1].memories

    def __delete(self, key: MemKey) -> None:
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

    def __insert_leaf(self, leaf: Node, insert_key: MemKey, insert_val: MemVal, weight:float) -> None:

        assert leaf.is_leaf

        if leaf.n < self.splitter(self.root.n):

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

            self.splitting = True

            split_memories = leaf.memories
            split_keys     = list(split_memories.keys())

            new_parent          = leaf
            new_parent.left     = EMT_PY.Node(next(self.node_ids), new_parent)
            new_parent.right    = EMT_PY.Node(next(self.node_ids), new_parent)
            new_parent.g        = self.router_factory.create(split_keys)
            new_parent.memories = dict()

            for split_key in split_keys:
                self.__update_router(new_parent, split_key, split_memories[split_key], new_parent.g.predict(split_key), "insert",1)
                leaf = new_parent.left if new_parent.g.predict(split_key) < 0 else new_parent.right
                leaf.n+=1
                leaf.memories[split_key] = split_memories[split_key]
                self.leaf_by_key[split_key] = leaf

            direction = new_parent.g.predict(insert_key) or self.rng.choice([-1,1])
            mode      = "insert"
            self.__update_router(new_parent, insert_key, insert_val, direction, mode, weight)
            leaf = new_parent.left if direction < 0 else new_parent.right
            self.__insert_leaf(leaf, insert_key, insert_val, weight)
        
            self.splitting = False

    def __bounded(self, key: MemKey) -> None:
        for item in self.bounder.touch(key): 
            self.__delete(item)

    def __reroute(self) -> None:

        if self.rerouting: return

        flt_part = self.d - int(self.d)
        int_part = int(self.d)

        for _ in range(int_part + int(self.rng.random() < flt_part)):

            x = self.rng.choice(list(self.leaf_by_key.keys()))
            o = self.leaf_by_key[x].memories[x]

            self.rerouting = True

            old_n = self.root.n

            self.__delete(x)
            assert self.root.n == old_n-1

            self.learn(x, o, 1)
            assert self.root.n == old_n

            self.rerouting = False

    def __path(self, key: MemKey, node: Node) -> Iterable[Node]:
        yield node
        while not node.is_leaf:
            label = node.g.predict(key) or self.rng.choice([-1,1])
            node  = node.left if label < 0 else node.right 
            yield node

    def __query(self, key: MemKey, init: Node) -> Tuple[MemKey, MemVal, MemScore]:

        final_node = list(self.__path(key, init))[-1]
        assert (final_node.is_leaf and final_node.memories) or self.splitting
        
        memories = final_node.memories

        if not memories:
            top_mem_key   = None
            top_mem_val   = None
            top_mem_score = None
        else:
            mem_keys     = list(memories.keys())
            mem_scores   = self.scorer.predict(key, mem_keys)
            tie_breakers = self.rng.randoms(len(mem_scores)) #randomly break
            sorted_mems  = list(sorted(zip(mem_scores, tie_breakers, mem_keys)))

            top_mem_key   = sorted_mems[0][2]
            top_mem_val   = memories[top_mem_key]
            top_mem_score = sorted_mems[0][0]

        return top_mem_key, top_mem_val, top_mem_score

    def __update_router(self, node: Node, key: MemKey, val: MemVal, direction: float, mode:str, weight: float) -> None:

        assert not node.is_leaf

        #direction is no longer used because we query the left and right
        #val       is no longer used because we use the score from the left and the right

        if not isinstance(self.router_factory,EigenRouter):
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

        if isinstance(self.router_factory, EigenRouter) or (isinstance(self.router_factory, VowpRouter) and self.router_factory._fixed): return

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

        if leaf.memories:
            mem_key_err_pairs = [ (k, (val-v)**2) for k,v in leaf.memories.items() ]
            mem_keys, mem_errs = zip(*mem_key_err_pairs)
            self.scorer.update(key, mem_keys, mem_errs, weight)

class EMT:

    def __init__(self, split:int = 100, scorer:int=3, router:int=2, bound:int=-1, interactions: Sequence[str]=[], rng : int = 1337) -> None:

        self._args = (split, scorer, router, bound, interactions, rng)

        vw_args = [
            "--eigen_memory_tree",
            f"--tree {bound}",
            f"--leaf {split}",
            f"--scorer {scorer}",
            f"--router {router}",
            "--min_prediction 0",
            "--max_prediction 3",
            "--coin",
            "--noconstant",
            f"--power_t {0}",
            "--loss_function squared",
            f"-b {26}",
            "--initial_weight 0",
            *[ f"--interactions {i}" for i in interactions ]
        ]

        init_args = f"{' '.join(vw_args)} --quiet --random_seed {rng}"
        label_type = 2

        self._vw = VowpalMediator().init_learner(init_args, label_type)

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (EMT, self._args)
    
    @property
    def params(self) -> Dict[str,Any]:
        keys = ['split', 'scorer', 'router', 'bound', 'X']
        return { 'type':'EMT', **dict(zip(keys,self._args))}

    def predict(self, key) -> Tuple[MemVal,MemScore]:
        ex = self._vw.make_example({'x': key.raw('x'), 'a': key.raw('a')}, None)
        pr = self._vw.predict(ex)
        cf = ex.get_confidence()
        return (pr,cf)

    def learn(self, key: MemKey, value: MemVal, weight: float):
        self._vw.learn(self._vw.make_example({'x': key.raw('x'), 'a': key.raw('a')}, f"{value} {weight}"))

class CMT:

    def __init__(self, n_nodes:int=100, leaf_multiplier:int=15, dream_repeats:int=5, alpha:float=0.5, coin:bool = True, interactions: Sequence[str]=[], rng : int = 1337) -> None:

        self._args = (n_nodes, leaf_multiplier, dream_repeats, alpha, coin, interactions, rng)

        vw_args = [
            f"--memory_tree {n_nodes}",
            "--learn_at_leaf",
            "--online 1",
            f"--leaf_example_multiplier {leaf_multiplier}",
            f"--dream_repeats {dream_repeats}",
            f"--alpha {alpha}",
            f"--power_t {0}",
            f"-b {25}",
            *[ f"--interactions {i}" for i in interactions ]
        ]

        if coin: vw_args.append("--coin")

        init_args = f"{' '.join(vw_args)} --quiet --random_seed {rng}"
        label_type = 2

        self._vw = VowpalMediator().init_learner(init_args, label_type)

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (CMT, self._args)

    @property
    def params(self) -> Dict[str,Any]:
        keys = ['nodes','multiplier','dreams','alpha','coin','X']
        return { 'type':'CMT', **dict(zip(keys,self._args)) }

    def predict(self, key) -> Tuple[MemVal,MemScore]:
        ex = self._vw.make_example({'x': key.raw('x'), 'a': key.raw('a')}, None)
        pr = self._vw.predict(ex)
        cf = ex.get_confidence()
        return (pr,cf)

    def learn(self, key: MemKey, value: MemVal, weight: float):
        self._vw.learn(self._vw.make_example({'x': key.raw('x'), 'a': key.raw('a')}, f"{value} {weight}"))
