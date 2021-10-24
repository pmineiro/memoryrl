from vowpalwabbit import pyvw
from examples import IdentityExample

bits = 20

class Logistic_VW:

    class Logistic_VW_Router:

        def __init__(self, vw, index):
            self.vw = vw
            self.exampler = IdentityExample(int(index*2**bits))

        def predict(self, xraw):
            return self.vw.predict(self.exampler.make_example(self.vw, xraw.features()))

        def update(self, xraw, y, w):
            self.vw.learn(self.exampler.make_example(self.vw, xraw.features(), 0, y, w))

        def __reduce__(self):
            return (Logistic_VW,())

    def __init__(self, power_t:float) -> None:
        #we add 20 to bits which means we can have 2**20 internal nodes
        self._power_t = power_t
        self._vw      = pyvw.vw(f'--quiet -b {bits+20} --loss_function logistic --noconstant --power_t {power_t} --link=glf1 --sparse_weights')
        self._index   = -1

    def __call__(self) -> Logistic_VW_Router:
        self._index += 1
        return Logistic_VW.Logistic_VW_Router(self._vw, self._index)
        
    def __repr__(self) -> str:
        return f"vw(power_t={self._power_t})"
    
    def __str__(self) -> str:
        return self.__repr__()

    def __reduce__(self):
        return (type(self),(self._power_t,))

class Logistic_SK:

    class Logistic_SK_Router:
        def __init__(self):

            from sklearn.linear_model import SGDClassifier
            self.clf  = SGDClassifier(loss="log", average=True, learning_rate='constant', eta0=0.5)
            self.is_fit = False

        def predict(self, x):
            return 1 if not self.is_fit else self.clf.predict(self._domain(x))[0]

        def update(self, x, y, w):
            self.clf.partial_fit(self._domain(x), [y], sample_weight=[w], classes=[-1,1])
            self.is_fit = True

        def _domain(self, x):
            return x.features()
    
    def __call__(self) -> Logistic_SK_Router:
        return Logistic_SK.Logistic_SK_Router()

    def __repr__(self) -> str:
        return f"sk"
    
    def __str__(self) -> str:
        return self.__repr__()
