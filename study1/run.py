from learners import ResidualLearner_1, ResidualLearner_2, JordanLogisticLearner, JordanVowpalLearner

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, UcbBanditLearner, CorralLearner
from coba.simulations import OpenmlSimulation

experiment       = "test"
processes        = 1
maxtasksperchild = None
seed             = 10

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}.log"

learners = [
    JordanVowpalLearner(0.1, 1, 200),
    JordanLogisticLearner(0.1, 1, 200),
    ResidualLearner_1(0.1, 200, 0.50),
    ResidualLearner_1(0.1, 200, 0.75),
    VowpalLearner(epsilon=0.1, seed=seed),
    UcbBanditLearner(),
    CorralLearner([ResidualLearner_2(0.1, 200), VowpalLearner(epsilon=0.1,seed=seed)], eta=.075, T=4000)
]

if __name__ == '__main__':
    Benchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).ignore_raise(False).evaluate(learners, log, seed).standard_plot()