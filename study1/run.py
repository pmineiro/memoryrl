from learners import ResidualLearner_1, ResidualLearner_2, JordanLogisticLearner, JordanVowpalLearner, MemorizedLearner_1
from sources import MediamillSource

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, UcbBanditLearner, CorralLearner
from coba.tools import CobaRegistry

CobaRegistry.register("Mediamill", MediamillSource)

experiment       = "media"
processes        = None
maxtasksperchild = None
seed             = 10
ignore_raise     = False

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}.log"

learners = [
    ResidualLearner_1(0.1, 200),
    MemorizedLearner_1(0.1, 200),
    UcbBanditLearner(),
    VowpalLearner(epsilon=0.1, seed=seed),
    CorralLearner([ResidualLearner_2(0.1, 200), VowpalLearner(epsilon=0.1,seed=seed)], eta=.075, T=4000)
]

if __name__ == '__main__':
    Benchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).ignore_raise(ignore_raise).evaluate(learners, log, seed).standard_plot()