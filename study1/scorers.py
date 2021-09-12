from coba.random import CobaRandom
from vowpalwabbit import pyvw

import scipy.sparse as sp
import numpy as np

bits = 20

#DirectDistScorer
#DiffDistScorer
#DirectLogitScorer
#DiffLogitScorer

#Base can be added to predict and learn... no need to process it separately
#Predict "0 0 base | features" where "0 0" are ignored by vw
#Learn   "1 1 base | features" where "1 1" are not ignored by vw

class RegrScorer:

    def __init__(self, l2=0, power_t=0, base="none"):

        assert base in ["none", "reward", "dist"]

        ignore_linear = "--ignore_linear a --ignore_linear b --ignore_linear c --ignore_linear d"
        interactions = "--interactions ac --interactions ad --interactions bc --interactions bd --interactions abcd"

        self.vw = pyvw.vw(f'--quiet -b {bits} --l2 {l2} --power_t {power_t} --min_prediction -1 --max_prediction 2' + " " + ignore_linear + " " + interactions )

        self.t       = 0
        self.l2      = l2
        self.power_t = power_t
        self.base    = base

        self.rng = CobaRandom(1)

        self.args = (self.l2, self.power_t, self.base)

    def __reduce__(self):
        return (type(self), self.args)

    @property
    def params(self):
        return ('regr',) + self.args

    def _vw_featurize(self, features):
        
        if isinstance(features, dict):
            final_features = features.items()
        else:
            final_features = list(zip(map(str,range(len(features))), features))

            final_final_features = []

            for k,v in final_features:
                if isinstance(v, str):
                    #add underscore to reduce collision risks
                    final_final_features.append(("_"+v,1))
                else:
                    final_final_features.append((k,v))
            
            return final_final_features

    def _base(self, query, memory, reward):
        if self.base == "none":
            return 0

        if self.base == "reward":
            return reward

        if self.base == "dist":
            ef1 = query.features()
            ef2 = memory.features()

            if isinstance(ef1,sp.spmatrix):
                return 1 - (ef1-ef2).power(2).sum() / ef1.nnz
            else:
                return 1 - ((ef1-ef2)**2).sum() / ef1.shape[1]

    def example(self, query, memory, base=0, label = 0, weight=1):

        # query (a,b), memory (c,d)

        a = self._vw_featurize(query.context)
        b = self._vw_featurize(query.action)
        c = self._vw_featurize(memory.context)
        d = self._vw_featurize(memory.action)

        ex = pyvw.example(self.vw, {"a": a, "b": b, "c":c, "d":d })
        ex.set_label_string(f"{label} {weight} {base}")

        return ex

    def predict(self, xraw, z):

        return self.vw.predict(self.example(xraw, z[0], self._base(xraw, z[0], z[1])))

    def update(self, xraw, zs, r):

        self.t += 1
        
        self.vw.learn(self.example(xraw, zs[0][0], self._base(xraw, zs[0][0], zs[0][1]), r))

class ClassScorer:

    def __init__(self, l2=0, power_t=0, type="direct"):

        assert type in ["direct", "difference"]

        self.vw = pyvw.vw(f'--quiet -b {bits} --l2 {l2} --power_t {power_t} --noconstant --loss_function logistic --link=glf1')

        self.t       = 0
        self.l2      = l2
        self.power_t = power_t
        self.type    = type

        self.rng = CobaRandom(1)

        self.args = (self.l2, self.power_t, type)

    def __reduce__(self):
        return (type(self), self.args)

    @property
    def params(self):
        return ('class',) + self.args

    def example(self, vw, feats, label = 0, weight = 0):

        if isinstance(feats, sp.spmatrix):
            keys = list(map(str,feats.indices))
            vals = feats.data
        else:
            keys = list(map(str,range(feats.shape[1])))
            vals = feats[0]

        ex = pyvw.example(vw, {"x": list(zip(keys, vals))})
        ex.set_label_string(f"{label} {weight}")

        return ex

    def feat(self, ex1, ex2):

        ef1 = ex1.features()
        ef2 = ex2.features()

        if isinstance(ef1,sp.spmatrix):
            return (ef1-ef2).power(2)
        else:
            return (ef1-ef2)**2

    def predict(self, xraw, z):

        feats = self.feat(xraw, z[0])

        return self.vw.predict(self.example(self.vw, feats))

    def update(self, xraw, zs, r):

        self.t += 1

        # Label is final reward
        # Base is 1-normalized distance
        # Weight is 1/probability

        # query (a,b), memory (c,d)

        # ac bd abcd

        #zs[0] is closet to observation
        #zs[1] is furthest from observation

        weight = r

        if len(zs) == 1:
            feats = self.feat(xraw, zs[0][0])
            self.vw.learn(self.example(self.vw, feats, 1, weight))
        
        if len(zs) == 2:
            f1 = self.feat(xraw, zs[0][0])
            f2 = self.feat(xraw, zs[1][0])

            if self.type == "direct":
                self.vw.learn(self.example(self.vw, f1,  1, weight))
                self.vw.learn(self.example(self.vw, f2, -1, weight))

            if self.type == "difference":
                self.vw.learn(self.example(self.vw, f1-f2, 1, weight))

class DistanceScorer:

    def __init__(self, order=2, norm="mean"):

        assert norm in ['max','mean']

        self.i     = 0
        self.stat  = norm
        self.order = order
        self.norm  = 1

    @property
    def params(self):
        return ('distance', self.order, self.stat)


    def distance(self,x1,x2):
        diff = x1.features()-x2.features()

        if isinstance(diff,sp.spmatrix):
            return np.linalg.norm(diff.data, ord=self.order)**2
        else:
            return np.linalg.norm(diff     , ord=self.order)**2

    def predict(self, xraw, z):
        return -self.distance(xraw, z[0]) / (self.norm if self.norm != 0 else 1)

    def update(self, xraw, zs, r):

        for z in zs:
            self.i += 1
            new_observation = self.distance(xraw, z[0])

            if self.stat == 'max':
                self.norm = max(self.norm, new_observation)

            if self.stat == 'mean':
                self.norm = (1-1/self.i) * self.norm + (1/self.i) * new_observation

class AdditionScorer:

    def __init__(self, scorers):
        self.scorers = scorers

    def predict(self, xraw, z):
        return sum([scorer.predict(xraw,z) for scorer in self.scorers])

    def update(self, xraw, zs, r):
        for scorer in self.scorers:
            scorer.update(xraw,zs,r)

    @property
    def params(self):
        return ('addition', [ scorer.params for scorer in self.scorers])

class DiffSubScorer:

    def __init__(self, l2=0, power_t=0.25, dist='max'):

        from vowpalwabbit import pyvw

        self.vw1 = pyvw.vw(f'--quiet -b {bits} --l2 {l2} --power_t {power_t} --noconstant --loss_function logistic --link=glf1')
        self.vw2 = pyvw.vw(f'--quiet -b {bits} --l2 {l2} --power_t {power_t} --loss_function quantile --max_prediction 2 --min_prediction -2')

        self.power_t = power_t
        self.l2      = l2

        self.t        = 0
        self.dist     = dist
        self.max_dist = 1

    def example(self,feat,vw):
        if isinstance(feat, sp.spmatrix):
            ns   = "x" 
            keys = list(map(str,feat.indices))
            data = feat.data
        else:
            ns   = "x"
            keys = list(map(str,range(feat.shape[1])))
            data = feat[0]

        return pyvw.example(vw, {ns: list(zip(keys, data))})

    def feat(self, ex1, ex2s):
        
        ef1 = ex1.features()

        final_feat = None

        for ex2 in ex2s:

            ef2  = ex2[0].features()
            
            if isinstance(ef1,sp.spmatrix):
                feat = (ef1-ef2).power(2)
            else:
                feat = (ef1-ef2)**2
            
            final_feat = feat if final_feat is None else final_feat - feat

        return final_feat

    def predict(self, xraw, z):

        feat = self.feat(xraw, [z])
        dist = feat.sum()
        pred = self.vw1.predict(self.example(feat,self.vw1))

        if self.dist == 'none':
            return pred
        elif self.dist == 'scaled':
            return pred-.01*dist
        elif self.dist == 'max':
            return pred-dist/self.max_dist
        elif self.dist == 'lin':
            return pred - self.vw2.predict(self.example(feat, self.vw2))
        else:
            return pred - 0.01 * self.vw2.predict(self.example(feat, self.vw2))

    def update(self, xraw, z, r):

        self.t += 1

        #increase distance between z[0] and x
        #decrease distance between z[1] and x

        #||z[0]-x||^2 - ||z[1]-x||^2

        if len(z) == 1:
            self.max_dist = max(self.max_dist, self.feat(xraw, z).sum())

        if len(z) == 2:
            feat = self.feat(xraw, z)
            example = self.example(feat, self.vw1)
            example.set_label_string(f"1 {r}")
            self.vw1.learn(example)

        if len(z) == 2:
            example = self.example(feat, self.vw2)
            example.set_label_string(f"-1 {r}" )
            self.vw2.learn(example)

    @property
    def params(self):
        return { 'st': 'vw1', 'power_t': self.power_t, 'l2': self.l2, 'dist': self.dist }

    def __reduce__(self):
        return (type(self), (self.l2, self.power_t, self.dist))

class LearnedCos_SK:

    def __init__(self,learn=True):

        from sklearn.linear_model import SGDRegressor

        self.clf    = SGDRegressor(loss="squared_loss")
        self.is_fit = False
        self.learn  = learn

    def example_features(self, ex):
        return ex.features()

    def feat_init(self, ex1, ex2s):
        # should return 0 as best
        ef1 = self.example_features(ex1)

        final_feat = None
        final_init = None

        for ex2 in ex2s:

            ef2 = self.example_features(ex2)
            
            if isinstance(ef1,sp.spmatrix):
                nnz = len(set(ef1.indices) | set(ef1.indices))

                mf1 = ef1.data.sum() / nnz
                mf2 = ef2.data.sum() / nnz

                sf1 = ((ef1.data**2).sum()/nnz-mf1**2)**(1/2)
                sf2 = ((ef2.data**2).sum()/nnz-mf2**2)**(1/2)
                
                feat = (ef1.multiply(ef2) - ef1.multiply(mf2) - ef2.multiply(mf1)).multiply(1/(nnz*sf1*sf2))
                init = 1-feat.data.sum() + mf1*mf2/(sf1*sf2)
            else:
                nnz = ef1.shape[1]

                mf1 = ef1.sum() / nnz
                mf2 = ef2.sum() / nnz

                sf1 = ((ef1**2).sum()/nnz-mf1**2)**(1/2)
                sf2 = ((ef2**2).sum()/nnz-mf2**2)**(1/2)
                
                feat = (ef1 * ef2 - mf2*ef1 - mf1*ef2)/(nnz*sf1*sf2)
                init = 1-feat.sum() + mf1*mf2/(sf1*sf2)

            final_feat = feat if final_feat is None else final_feat + feat
            final_init = init if final_init is None else final_init + init

        return final_feat,final_init/len(ex2s)

    def predict(self, trigger, memory):

        feat,init = self.feat_init(trigger,[memory[0]]) 

        pred = 0 if not self.is_fit else self.clf.predict(feat)[0]

        return 1-(init+pred)

    def update(self, trigger, memories, reward):

        if not self.learn: return

        feat,init = self.feat_init(trigger,[m[0] for m in memories])
        self.is_fit = True
        self.clf.partial_fit(feat, [1-reward-init])

class LearnedCorr_SK:

    def __init__(self,learn=True):

        from sklearn.linear_model import SGDRegressor

        self.clf    = SGDRegressor(loss="squared_loss")
        self.is_fit = False
        self.learn  = learn

    def example_features(self, ex):
        return ex.features()

    def feat_init(self, ex1, ex2s):
        # should return 0 as best
        ef1 = self.example_features(ex1)

        final_feat = None
        final_init = None

        for ex2 in ex2s:

            ef2 = self.example_features(ex2)
            
            if isinstance(ef1,sp.spmatrix):
                nnz = len(set(ef1.indices) | set(ef1.indices))

                mf1 = ef1.data.sum() / nnz
                mf2 = ef2.data.sum() / nnz

                sf1 = ((ef1.data**2).sum()/nnz-mf1**2)**(1/2)
                sf2 = ((ef2.data**2).sum()/nnz-mf2**2)**(1/2)
                
                feat = (ef1.multiply(ef2) - ef1.multiply(mf2) - ef2.multiply(mf1)).multiply(1/(nnz*sf1*sf2))
                init = 1-feat.data.sum() + mf1*mf2/(sf1*sf2)
            else:
                nnz = ef1.shape[1]

                mf1 = ef1.sum() / nnz
                mf2 = ef2.sum() / nnz

                sf1 = ((ef1**2).sum()/nnz-mf1**2)**(1/2)
                sf2 = ((ef2**2).sum()/nnz-mf2**2)**(1/2)
                
                feat = (ef1 * ef2 - mf2*ef1 - mf1*ef2)/(nnz*sf1*sf2)
                init = 1-feat.sum() + mf1*mf2/(sf1*sf2)

            final_feat = feat if final_feat is None else final_feat + feat
            final_init = init if final_init is None else final_init + init

        return final_feat,final_init/len(ex2s)

    def predict(self, trigger, memory):

        feat,init = self.feat_init(trigger,[memory[0]]) 

        pred = 0 if not self.is_fit else self.clf.predict(feat)[0]

        return 1-(init+pred)

    def update(self, trigger, memories, reward):

        if not self.learn: return

        feat,init = self.feat_init(trigger,[m[0] for m in memories])
        self.is_fit = True
        self.clf.partial_fit(feat, [1-reward-init])
