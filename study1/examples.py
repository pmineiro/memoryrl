from abc import abstractmethod
from vowpalwabbit import pyvw
import scipy.sparse as sp

class MemExample:
    
    @abstractmethod
    def interactions(self):
        pass

    @abstractmethod
    def ignored(self):
        pass
    
    @abstractmethod
    def make_example(self, vw, query, memory, base=0, label=0, weight=1):
        pass

class IdentityExample(MemExample):

    def _vw_featurize(self, ns, features):
        
        if isinstance(features, sp.spmatrix):
            keys = list(map(str,features.indices))
            vals = features.data
            return list(zip(keys,vals))
        elif isinstance(features, dict):
            return [ (k,v) for k,v in features.items() if v != 0 ]
        else:

            final_features = list(zip(map(str,range(len(features[0]))), features[0]))

            final_final_features = []

            for k,v in final_features:
                if v == 0: continue

                if isinstance(v, str):
                    final_final_features.append((ns+"_"+v,1))
                else:
                    final_final_features.append((ns+"_"+k,v))
            
            return final_final_features

    def interactions(self):
        return []

    def ignored(self):
        return []

    def make_example(self, vw, features, base=None, label=None, weight=None):

        x = self._vw_featurize("x", features)

        ex = pyvw.example(vw, {"x": x })

        if label is not None:
            ex.set_label_string(f"{label} {weight} {base}")

        return ex

    def __repr__(self) -> str:
        return "identity"

    def __str__(self) -> str:
        return "identity"

class InteractionExample(MemExample):

    def __init__(self, interactions=["ac","ad","bc","bd","abcd"], ignored=["a","b","c","d"]):
        self._interactions = interactions
        self._ignored      = ignored

    def interactions(self):
        return self._interactions

    def ignored(self):
        return self._ignored

    def _vw_featurize(self, ns, features):
        
        if isinstance(features, dict):
            final_features = features.items()
        else:
            final_features = list(zip(map(str,range(len(features))), features))

            final_final_features = []

            for k,v in final_features:
                if isinstance(v, str):
                    final_final_features.append((ns+"_"+v,1))
                else:
                    final_final_features.append((ns+"_"+k,v))
            
            return final_final_features

    def make_example(self, vw, query, memory, base=0, label=0, weight=1):

        a = self._vw_featurize("a", query.context)
        b = self._vw_featurize("b", query.action)
        c = self._vw_featurize("c", memory.context)
        d = self._vw_featurize("d", memory.action)

        ex = pyvw.example(vw, {"a": a, "b": b, "c":c, "d":d })
        ex.set_label_string(f"{label} {weight} {base}")

        return ex

    def __repr__(self) -> str:
        return f"pure{(self._interactions, self._ignored)}"

    def __str__(self) -> str:
        return self.__repr__()

class DifferenceExample(MemExample):

    def __init__(self, element_wise_op:str="^2") -> None:

        assert element_wise_op in ["^2", "abs", "none"]

        self._element_wise_op = element_wise_op
 
    def interactions(self):
        return []

    def ignored(self):
        return []

    def feat(self, ex1, ex2):

        ef1 = ex1.features()
        ef2 = ex2.features()

        diff = ef1-ef2

        if self._element_wise_op == "none":
            return diff
        
        if self._element_wise_op == "^2" and sp.issparse(diff):
            return diff.power(2) #fastest way to calculate this that I could find

        if self._element_wise_op == "^2" and not sp.issparse(diff):
            return diff**2

        if self._element_wise_op == "abs":
            return abs(diff)

        raise Exception("something unexpected happened")

    def make_example(self, vw, query, memory, base=0, label=0, weight=1):

        feats = self.feat(query, memory)

        if isinstance(feats, sp.spmatrix):
            keys = list(map(str,feats.indices))
            vals = feats.data
        else:
            keys = list(map(str,range(feats.shape[1])))
            vals = feats[0]

        ex = pyvw.example(vw, {"x": list(zip(keys, vals))})
        ex.set_label_string(f"{label} {weight} {base}")

        return ex
    
    def __repr__(self) -> str:
        return f"diff({self._element_wise_op})"

    def __str__(self) -> str:
        return self.__repr__()