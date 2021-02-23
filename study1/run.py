from learners import ResidualLearner_1, ResidualLearner_2, JordanLogisticLearner, JordanVowpalLearner

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, UcbBanditLearner, CorralLearner

experiment       = "medish"
processes        = 2
maxtasksperchild = None
seed             = 10
ignore_raise     = True

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}.log"

learners = [
    JordanVowpalLearner(0.1, 0.1, 200),
    JordanVowpalLearner(0.1, 1, 200),
    ResidualLearner_1(0.1, 200),
    VowpalLearner(epsilon=0.1, seed=seed),
    UcbBanditLearner(),
    CorralLearner([ResidualLearner_2(0.1, 200), VowpalLearner(epsilon=0.1,seed=seed)], eta=.075, T=4000)
]

if __name__ == '__main__':
    Benchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).ignore_raise(ignore_raise).evaluate(learners, log, seed).standard_plot()