from abc import abstractmethod
from vowpalwabbit import pyvw
import scipy.sparse as sp

class Exampler:

    @abstractmethod
    def make_example(self, vw, query, memory, base=0, label=0, weight=1):
        pass

class IdentityExampler(Exampler):

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

class PureExampler(Exampler):

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
        return "pure"

    def __str__(self) -> str:
        return "pure"

class DiffSquareExampler(Exampler):

    def feat(self, ex1, ex2):

        ef1 = ex1.features()
        ef2 = ex2.features()

        if isinstance(ef1, sp.spmatrix):
            return (ef1-ef2).power(2)
        else:
            return (ef1-ef2)**2

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
        return "diffsquare"

    def __str__(self) -> str:
        return "diffsquare"