from learners import ResidualLearner, MemorizedLearner, CorralEnsemble

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, UcbBanditLearner


experiment       = "madish"
processes        = 1
maxtasksperchild = None
seed             = 10
ignore_raise     = False

max_memories = 2000
learn_distance = True
epsilon = 0.1

json = f"./study1/finals/small_action/madish.json"
log  = f"./study1/finals/small_action/madish.log"

learners = [
    MemorizedLearner(epsilon , max_memories, learn_distance),
    ResidualLearner (epsilon , max_memories, learn_distance),
    VowpalLearner(epsilon=epsilon, seed=seed),
    UcbBanditLearner(),
    CorralEnsemble(max_memories, epsilon, eta=.075, fix_count=3000, T=4000, seed=seed),
]

if __name__ == '__main__':
    Benchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).ignore_raise(ignore_raise).evaluate(learners, log, seed).standard_plot()