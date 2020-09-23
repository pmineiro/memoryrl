import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max', type=int, default=1000)
parser.add_argument('--pruneStrat', type=str, default='')
args = parser.parse_args()

class FOC:
    class EasyAcc:
        def __init__(self):
            self.n = 0
            self.sum = 0
            
        def __iadd__(self, other):
            self.n += 1
            self.sum += other
            return self
            
        def mean(self):
            return self.sum / max(self.n, 1)
 
    import torch
    class LogisticRegressor(torch.nn.Module):        
        def __init__(self, input_dim, output_dim, eta0):
            import torch
            
            super(FOC.LogisticRegressor, self).__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)
            self.loss = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=eta0)
            self.eta0 = eta0
            self.n = 0
            
        def forward(self, X):
            import numpy as np
            import torch

            return self.linear(torch.autograd.Variable(torch.from_numpy(X)))
        
        def predict(self, X):
            import torch
            
            return torch.argmax(self.forward(X), dim=1).numpy()
        
        def set_lr(self):
            from math import sqrt
            lr = self.eta0 / sqrt(self.n)
            for g in self.optimizer.param_groups:
                g['lr'] = lr

        def partial_fit(self, X, y, sample_weight=None, **kwargs):
            import torch
            
            self.optimizer.zero_grad()
            yhat = self.forward(X)
            if sample_weight is None:
                loss = self.loss(yhat, torch.from_numpy(y))
            else:
                loss = torch.from_numpy(sample_weight) * self.loss(yhat, torch.from_numpy(y))
            loss.backward()
            self.n += X.shape[0]
            self.set_lr()
            self.optimizer.step() 

    class LogisticModel:
        def __init__(self, *args, **kwargs):
            kwargs['output_dim'] = 2
            self.model = FOC.LogisticRegressor(*args, **kwargs)
            
        def predict(self, x):
            import numpy as np
            
            F = self.model.forward(X=np.array([x], dtype='float32')).detach().numpy()
            dF = F[:,1] - F[:,0]
            return -1 + 2 * dF          
        
        def update(self, x, y, w):
            import numpy as np
            
            assert y == 1 or y == -1
            
            self.model.partial_fit(X=np.array([x], dtype='float32'), 
                                   y=(1 + np.array([y], dtype='int')) // 2, 
                                   sample_weight=np.array([w], dtype='float32'),
                                   classes=(0, 1))

    class LearnedEuclideanDistance:
        def __init__(self, *args, **kwargs):
            kwargs['output_dim'] = 2
            self.model = FOC.LogisticRegressor(*args, **kwargs)
            self.model.linear.weight.data[0,:].fill_(0.01 / kwargs['input_dim'])
            self.model.linear.weight.data[1,:].fill_(-0.01 / kwargs['input_dim'])
            self.model.linear.bias.data.fill_(0.0)
            self.model.linear.bias.requires_grad = False
        
        def predict(self, x, z):
            import numpy as np
            
            (xprime, omegaprime) = z
            
            dx = np.array([x], dtype='float32')
            dx -= [xprime]
            dx *= dx
            
            F = self.model.forward(dx).detach().numpy()
            dist = F[0,1] - F[0,0]
            return dist
        
        def update(self, x, z, r):
            import numpy as np
            
            if r == 1 and len(z) > 1 and z[0][1] != z[1][1]:
                dx = np.array([ z[0][0], z[1][0] ], dtype='float32')
                dx -= [x]
                dx *= dx
                y = np.array([1, 0], dtype='int')    
                self.model.partial_fit(X=dx,
                                       y=y,
                                       sample_weight=None, # (?)
                                       classes=(0, 1))
            
    class EuclideanDistance:
        def __init__(self):
            pass
        
        def predict(self, x, z):
            import numpy as np
            from math import sqrt
            
            (xprime, omegaprime) = z
            
            return -np.linalg.norm(np.array(x) - np.array(xprime))
        
        def update(self, x, y, w):
            pass
            
    def doit():
        from collections import Counter
        from sklearn.datasets import fetch_covtype
        from sklearn.decomposition import PCA
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import accuracy_score
        import pdb
        from math import ceil
        import numpy as np
        import random
        import torch
        from cmt import CMT

        cov = fetch_covtype()
        cov.data = PCA(whiten=True).fit_transform(cov.data)
        classes = np.unique(cov.target - 1)
        print(Counter(cov.target - 1))
        ndata = len(cov.target)
        order = np.random.RandomState(seed=42).permutation(ndata)
        ntrain = ceil(0.9 * ndata)
        Object = lambda **kwargs: type("Object", (), kwargs)()
        train = Object(data = cov.data[order[:ntrain]], target = cov.target[order[:ntrain]] - 1)
        test = Object(data = cov.data[order[ntrain:]], target = cov.target[order[ntrain:]] - 1)
        
        input_dim = train.data[0].shape[0]
        routerFactory = lambda: FOC.LogisticModel(eta0=0.1, input_dim=input_dim)
        scorer = FOC.LearnedEuclideanDistance(eta0=1e-4, input_dim=input_dim)
        randomState = random.Random()
        #randomState.seed(45)
        #torch.manual_seed(2112)

  
        #print(args.pruneStrat)
        #print('{:8.8s}\t{:8.8s}\t{:10.10s}\t{:10.10s}'.format(
        #    'n', 'emp loss', 'since last', 'last pred')
        #)



        for pno in range(5):
            print('new', flush=True)
            loss = FOC.EasyAcc()
            sincelast = FOC.EasyAcc()
            cmt = CMT(routerFactory=routerFactory, scorer=scorer, alpha=0.25, c=10, d=1, randomState=randomState,
                maxMemories=args.max, pruneStrat=args.pruneStrat)
            order = np.random.permutation(len(train.data))
            for n, ind in enumerate(order):
                t = train.data[ind]
                x = tuple(t)
                actual = train.target[ind]
                
                if n == 0:
                    pred = 0
                else:
                    u, z = cmt.query(x, k=1, epsilon=0.0)
                    pred = z[0][1] if len(z) else 0
                   
                loss += 0 if pred == actual else 1
                sincelast += 0 if pred == actual else 1
                
                if (n % 100 == 0): # and n & 0xAAAAAAAA == 0):
                    print('{:<8d}\t{:<8.3f}\t{:<10.3f}\t{:<10d}'.format(
                               loss.n, loss.mean(), sincelast.mean(), len(cmt.allkeys)),
                          flush=True)

                sincelast = FOC.EasyAcc()
                if n > 0:
                    u, z = cmt.query(x, k=2, epsilon=1.0)
                    if len(z):
                        r = 1 if z[0][1] == actual else -1
                        cmt.update(u, x, z, r)
                cmt.insert(x, actual)
                if n >= 10000: break

            sincelast = FOC.EasyAcc()

            
                        
def flass():
    import timeit
    print(timeit.timeit(FOC.doit, number=1))
    
flass()
