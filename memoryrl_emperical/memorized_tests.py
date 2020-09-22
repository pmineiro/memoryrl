import math
import numpy as np
from memorized import CMT

class CMTTests:
    class LinearModel:
        def __init__(self, *args, **kwargs):
            from sklearn import linear_model
            
            self.model = linear_model.SGDRegressor(*args, **kwargs)
            
        def predict(self, x):
            from sklearn.exceptions import NotFittedError 
            try:
                return self.model.predict(X=[x])[0]
            except NotFittedError:
                return 0
        
        def update(self, x, y, w):
            self.model.partial_fit(X=[x], y=[y], sample_weight=[w])
            
    class NormalizedLinearProduct:
        def __init__(self):
            pass
        
        def predict(self, x, z):            
            (xprime, omegaprime) = z
            
            xa = np.array(x)
            xprimea = np.array(xprime)
                        
            return np.inner(xa, xprimea) / math.sqrt(np.inner(xa, xa) * np.inner(xprimea, xprimea))
        
        def update(self, x, y, w):
            pass
 
    @staticmethod
    def displaynode(node, indent):
        if node is not None:
            from pprint import pformat
            print(indent, pformat((node, node.__dict__)))
            CMTTests.displaynode(node.left, indent + "*")
            CMTTests.displaynode(node.right, indent + "*")

    @staticmethod
    def displaytree(cmt):
        CMTTests.displaynode(cmt.root, indent="")

    @staticmethod
    def structureValid():
        import random
        
        routerFactory = lambda: CMTTests.LinearModel()
        scorer = CMTTests.NormalizedLinearProduct()
        randomState = random.Random()
        randomState.seed(2112)
        cmt = CMT(routerFactory=routerFactory, scorer=scorer, alpha=0.5, c=10, d=0, randomState=randomState)

        def checkNodeInvariants(node):
            assert node.parent is None or node.parent.left is node or node.parent.right is node
            assert node.left is None or node.n == node.left.n + node.right.n
            assert node.left is None or node.left.parent is node
            assert node.right is None or node.right.parent is node
            assert node.left is not None or node.n == len(node.memories)
    
        stuff = {}
        
        for _ in range(200):
            try:
                if stuff and randomState.uniform(0, 1) < 0.1:
                    # delete
                    x, omega = stuff.popitem()
                    cmt.delete(x)
                elif stuff and randomState.uniform(0, 1) < 0.1:
                    # query/update
                    somex = randomState.choice(list(stuff.keys()))
                    u, z = cmt.query(somex, 1, 0.1)
                    cmt.update(u, somex, z, randomState.uniform(0, 1))
                else:
                    # insert
                    x = tuple([ randomState.uniform(0, 1) for _ in range(3)])
                    omega = randomState.uniform(0, 1)
                    cmt.insert(x, omega)
                    stuff[x] = omega

                assert cmt.root.n == len(stuff)
                assert cmt.root.n == len(cmt.leafbykey)
                assert cmt.root.n == len(cmt.allkeys)
                assert cmt.root.n == len(cmt.allkeysindex)
                
                for z in stuff.keys():
                    assert z in cmt.leafbykey[z].memories
                    assert z in cmt.allkeysindex
                    assert cmt.allkeys[cmt.allkeysindex[z]] is z
                cmt.nodeForeach(checkNodeInvariants)
            except:
                print("--------------")
                CMTTests.displaytree(cmt)
                print("--------------")
                raise
                
        print('structureValid test pass')           
                       
    @staticmethod
    def selfconsistent():
        import random
        
        routerFactory = lambda: CMTTests.LinearModel()
        scorer = CMTTests.NormalizedLinearProduct()
        randomState = random.Random()
        randomState.seed(45)
        cmt = CMT(routerFactory=routerFactory, scorer=scorer, alpha=0.5, c=10, d=0, randomState=randomState)
        
        for _ in range(200):
            try:
                x = tuple([ randomState.uniform(0, 1) for _ in range(3)])
                omega = randomState.uniform(0, 1)

                cmt.insert(x, omega)
                u, [ (xprime, omegaprime) ] = cmt.query(x, k=1, epsilon=0)
                assert omega == omegaprime, '({}, [({}, {})]) = cmt.query({}) != {}'.format(u, xprime, omegaprime, x, omega)
            except:
                print("--------------")
                CMTTests.displaytree(cmt)
                print("--------------")
                raise
                
        print('selfconsistent test pass')
        
    @staticmethod
    def maxmemories():
        import random
        
        routerFactory = lambda: CMTTests.LinearModel()
        scorer = CMTTests.NormalizedLinearProduct()
        randomState = random.Random()
        randomState.seed(45)
        maxM = 100
        cmt = CMT(routerFactory=routerFactory, scorer=scorer, alpha=0.5, c=10, d=0, randomState=randomState, maxMemories=maxM)
        
        for _ in range(200):
            try:
                x = tuple([ randomState.uniform(0, 1) for _ in range(3)])
                omega = randomState.uniform(0, 1)

                cmt.insert(x, omega)
                assert len(cmt.leafbykey) <= maxM
            except:
                print("--------------")
                CMTTests.displaytree(cmt)
                print("--------------")
                raise
                
        print('maxmemories test pass')
       
    @staticmethod
    def all():
        CMTTests.structureValid()
        CMTTests.selfconsistent()
        CMTTests.maxmemories()

CMTTests().all()