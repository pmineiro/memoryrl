from learners import ResidualLearner, MemorizedLearner
from sources import MediamillSource, MemorizableSource

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, UcbBanditLearner, CorralLearner
from coba.tools import CobaRegistry

CobaRegistry.register("Mediamill", MediamillSource)
CobaRegistry.register("Memorizable", MemorizableSource)

experiment       = "test"
processes        = None
maxtasksperchild = None
seed             = 10
ignore_raise     = False

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}.log"

learners = [
    MemorizedLearner(0.1, 2000, learn_dist=True),
    MemorizedLearner(0.1, 2000, learn_dist=False),
    ResidualLearner(0.1, 2000, learn_dist=True),
    ResidualLearner(0.1, 2000, learn_dist=False),
    UcbBanditLearner(),
    VowpalLearner(epsilon=0.1, seed=seed),
    CorralLearner([ResidualLearner(0.1, 2000, True, "pct"), VowpalLearner(epsilon=0.1,seed=seed)], eta=.075, T=4000, seed=seed)
]

if __name__ == '__main__':
    Benchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).ignore_raise(ignore_raise).evaluate(learners, log, seed).standard_plot()