import copy
import bisect

from math        import log
from itertools   import count
from collections import Counter
from typing      import Dict, Any, Tuple, Hashable, Iterable, List

from routers   import RouterFactory, Router, ProjRouter, VowpRouter
from scorers   import Scorer
from splitters import Splitter

from coba.random import CobaRandom, shuffle

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
        return 1 + self.parent.depth if self.parent else 1

class EMT:

    def __init__(self,
        max_mem:int,
        router: RouterFactory,
        scorer: Scorer,
        c:Splitter,
        d:int,
        alpha:float=0.25,
        mlr:float = 0,
        max_depth = None,
        rng:int = 1337):

        self.max_mem   = max_mem
        self.g_factory = router
        self.f         = scorer
        self.alpha     = alpha
        self.c         = c
        self.d         = d
        self.rng       = CobaRandom(rng)
        self.node_ids  = iter(count())
        self.mlr       = mlr
        self.max_depth = max_depth

        self.root = Node(next(self.node_ids), None)
        self.leaf_by_key: Dict[MemKey,Node] = {}

        self.rerouting = False
        self.splitting = False

    @property
    def params(self) -> Dict[str,Any]:
        return { 'type':'CMT', 'm': self.max_mem, 'c': str(self.c), 'd': self.d, "a": self.alpha, "scr": str(self.f), "rou": str(self.g_factory) }

    def query(self, key) -> Tuple[MemVal,MemScore]:

        if self.root.n == 0: return [0,0]

        return self.__query(key, self.root)[1:3]

    def insert(self, key: MemKey, value: MemVal, weight: float):

        if key in self.leaf_by_key: return

        self.__update_omega(key, value)
        self.__update_scorer(list(self.__path(key,self.root))[-1], key, value, weight, mode="update")
        self.__update_routers(key, value, weight, mode="insert")
        self.__insert_leaf(list(self.__path(key,self.root))[-1], key, value, weight)
        self.__update_scorer(self.leaf_by_key[key], key, value, weight, mode="insert")
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

    def __update_omega(self, key: MemKey, outcome:float) -> None:
        if self.mlr > 0:
            top_key, top_val = self.__query(key,self.root)[0:2]
            self.leaf_by_key[top_key].memories[top_key] = self.mlr*outcome + (1-self.mlr)*top_val

    def __insert_leaf(self, leaf: Node, insert_key: MemKey, insert_val: MemVal, weight:float) -> None:

        assert leaf.is_leaf

        if leaf.n <= self.c(self.root.n) or leaf.depth == self.max_depth:

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
            new_parent.left     = Node(next(self.node_ids), new_parent)
            new_parent.right    = Node(next(self.node_ids), new_parent)
            new_parent.g        = self.g_factory.create(split_keys)
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

            self.insert(x, o, 1)
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
            mem_scores   = self.f.predict(key, mem_keys)
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

        if not isinstance(self.g_factory,ProjRouter):
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

        if isinstance(self.g_factory, ProjRouter) or (isinstance(self.g_factory, VowpRouter) and self.g_factory._fixed): return

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
            self.f.update(key, mem_keys, mem_errs, weight)

class DCI:

    class Index:
        def __init__(self, proj, features, keys, samples):

            import numpy as np
            import scipy.sparse as sp
            from sklearn.decomposition import TruncatedSVD

            self.features = tuple(features)
            raws2split    = [k.raw(self.features) for k in keys]
            is_sparse     = isinstance(keys[0].raw(self.features), dict)

            if is_sparse:
                all_mat2split  = sp.vstack([k.mat(self.features) for k in keys])
                rng_mat2split  = sp.vstack([k.mat(self.features) for k in shuffle(keys)[:50]])
            else:
                all_mat2split  = np.vstack([k.mat(self.features) for k in keys])
                rng_mat2split  = np.vstack([k.mat(self.features) for k in shuffle(keys)[:50]])

            if proj=="PCA":
                if is_sparse:
                    features = rng_mat2split
                    center   = sp.vstack([sp.csr_matrix(features.mean(axis=0))]*features.shape[0])
                else:
                    features = rng_mat2split
                    center   = np.vstack([features.mean(axis=0)]*features.shape[0])

                max_projector = TruncatedSVD(n_components=1).fit(features-center).components_.astype(float)[0]
            else:
                max_projector   = None
                max_dispersion  = 0

                if is_sparse:
                    indices = list(set(sp.find(rng_mat2split)[1]))
                else:
                    indices = list(range(len(raws2split[0])))

                for _ in range(samples):
                    projector = np.random.randn(len(indices))
                    projector = projector/np.linalg.norm(projector)

                    if is_sparse:
                        sparse_projector = np.zeros((rng_mat2split.shape[1]),float)
                        sparse_projector[indices] = projector
                        projector = sparse_projector

                    projections = projector @ rng_mat2split.T
                    dispersion  = projections.var()

                    if dispersion > max_dispersion:
                        max_projector   = projector
                        max_dispersion  = dispersion

            self._projector   = max_projector
            self._memories    = list(keys)
            self._projections = (max_projector @ all_mat2split.T).tolist()

        def insert(self, key):
            proj = (key.mat(self.features) @ self._projector)[0]
            index = bisect.bisect(self._projections, proj)

            self._memories.insert(index, key)
            self._projections.insert(index, proj)

        def query(self, key, n) -> Iterable[MemKey]:
            proj = (key.mat(self.features) @ self._projector)[0]
            index = bisect.bisect(self._projections, proj)

            left_idx  = index-1
            right_idx = index

            get = lambda i: abs(self._projections[i]-proj) if 0<=i and i<len(self._projections) else float('inf')

            left_val = get(left_idx)
            right_val = get(right_idx)

            mem_keys = []

            for _ in range(n):
                if left_val < right_val:
                    mem_keys.append(self._memories[left_idx])
                    left_idx -= 1
                    left_val = get(left_idx)
                elif left_val > right_val:
                    mem_keys.append(self._memories[right_idx])
                    right_idx += 1
                    right_val = get(right_idx)

            mem_keys.reverse()

            return mem_keys

    def __init__(self, n_index, n_top, n_samples, proj, scorer:Scorer, features = ['x'], rng:int = 1337) -> None:
        self._features  = features
        self._n_index   = n_index
        self._n_top     = n_top
        self._n_samples = n_samples
        self.f          = scorer
        self._proj      = proj
        
        self._projectors: List[DCI.Index] = []
        self._memories: Dict[MemKey,MemVal]  = {}
        self._index_mems: List[MemKey] = []

        self.rng = CobaRandom(rng)

    @property
    def params(self) -> Dict[str,Any]:
        return { 'type':'DCI', "n_top": self._n_top, "n_index": self._n_index, "n_samples": self._n_samples, "P": self._proj }

    def query(self, key: MemKey) -> Tuple[MemVal,MemScore]:

        memories = self.memories(key)

        if not memories:
            top_mem_key   = None
            top_mem_val   = None
            top_mem_score = None
        else:
            mem_keys     = list(memories.keys())
            mem_scores   = self.f.predict(key, mem_keys)
            tie_breakers = self.rng.randoms(len(mem_scores))
            sorted_mems  = list(sorted(zip(mem_scores, tie_breakers, mem_keys)))

            top_mem_key   = sorted_mems[0][2]
            top_mem_val   = memories[top_mem_key]
            top_mem_score = sorted_mems[0][0]

        return top_mem_val, top_mem_score

    def insert(self, key: MemKey, value: MemVal, weight: float):
        self._memories[key] = value
        self._index_mems.append(key)

        for projector in self._projectors:
            projector.insert(key)

        if len(self._index_mems) == self._n_index:
            self._projectors.append(DCI.Index(self._proj, self._features, self._index_mems, self._n_samples))
            for key in (self._memories.keys() - set(self._index_mems)):
                self._projectors[-1].insert(key)
            self._index_mems = []

        self.__update_scorer(key, value, weight, "insert")

    def update(self, key: MemKey, outcome: float, weight: float) -> None:
        self.__update_scorer(key, outcome, weight, "update")

    def memories(self, key: MemKey) -> Dict[MemKey,MemVal]:

        if self._projectors:
            votes = Counter()

            for projector in self._projectors:
                votes = votes + Counter(dict(zip(projector.query(key, self._n_top), count())))

            return { k: self._memories[k] for k,v in votes.most_common(100) }
        else:
            return self._memories

    def __update_scorer(self, key: MemKey, val: MemVal, weight:float, mode:str) -> None:

        assert mode in ['update','insert'] 

        mem_key_err_pairs = [ (k, (val-v)**2) for k,v in self.memories(key).items() ]

        if len(mem_key_err_pairs) == 0: return

        mem_keys, mem_errs = zip(*mem_key_err_pairs)
        self.f.update(key, mem_keys, mem_errs, weight)

class CMF:

    def __init__(self, n_trees:int, scorer:Scorer, tree) -> None:

        self._tree    = tree
        self._scorer  = scorer
        self._rng     = CobaRandom(1)
        self._trees   = [ copy.deepcopy(tree) for _ in range(n_trees)  ]

    @property
    def params(self) -> Dict[str,Any]:
        return { **self._tree.params, 'scr': str(self._scorer), 'type':'CMF', 'T': len(self._trees) }

    def query(self, key: MemKey) -> Tuple[MemVal,MemScore]:

        memories = self.memories(key)

        if not memories:
            top_mem_key   = None
            top_mem_val   = 0
            top_mem_score = 0
        else:
            mem_keys     = list(memories.keys())
            mem_scores   = self._scorer.predict(key, mem_keys)
            tie_breakers = self._rng.randoms(len(mem_scores)) #randomly break
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

    def memories(self, key: MemKey) -> Dict[MemKey,MemVal]:
        
        memories = {}
        votes = Counter()

        for tree in self._trees:
            tree_memories = tree.memories(key)
            memories.update(tree_memories)
            votes = votes + Counter(tree_memories.keys())

        return { k: memories[k] for k,v in votes.most_common(75) }

    def __update_scorer(self, key: MemKey, val: MemVal, weight:float, mode:str) -> None:

        assert mode in ['update','insert'] 

        mem_key_err_pairs = [ (k, (val-v)**2) for k,v in self.memories(key).items() ]

        if len(mem_key_err_pairs) == 0: return

        mem_keys, mem_errs = zip(*mem_key_err_pairs)
        self._scorer.update(key, mem_keys, mem_errs, weight)
