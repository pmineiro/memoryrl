from learners import ResidualLearner, MemorizedLearner, CorralRejectionLearner
from sources import MediamillSource, MemorizableSource

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, UcbBanditLearner
from coba.tools import CobaRegistry

CobaRegistry.register("Mediamill", MediamillSource)
CobaRegistry.register("Memorizable", MemorizableSource)

experiment       = "medish"
processes        = 1
maxtasksperchild = None
seed             = 10
ignore_raise     = False

max_memories = 2000
learn_distance = True

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}.log"

learners = [
    MemorizedLearner(0.1, max_memories, learn_distance),
    ResidualLearner(0.1, max_memories, learn_distance),
    UcbBanditLearner(),
    VowpalLearner(epsilon=0.1, seed=seed),
    CorralRejectionLearner([ResidualLearner(0.1, max_memories, learn_distance), MemorizedLearner(0.1, max_memories, learn_distance), VowpalLearner(epsilon=0.1,seed=seed)], eta=.075, fix_count=3000, T=4000, seed=seed),
    CorralRejectionLearner([ResidualLearner(0.1, max_memories, learn_distance), MemorizedLearner(0.1, max_memories, learn_distance), VowpalLearner(epsilon=0.1,seed=seed)], eta=.075, fix_count=None, T=4000, seed=seed),
]

if __name__ == '__main__':
    Benchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).ignore_raise(ignore_raise).evaluate(learners, log, seed).standard_plot()