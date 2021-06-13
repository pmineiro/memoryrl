import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import simulations #this import registers our simulations for use in experiments
from learners import ResidualLearner, MemorizedLearner, MemorizedIpsLearner, CorralEnsemble

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, UcbBanditLearner

experiment       = "madish"
processes        = 8
ignore_raise     = True

max_memories   = 3000
learn_distance = True
epsilon        = 0.1

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}.log"

learners = [
    VowpalLearner(epsilon=epsilon, seed=1),
    MemorizedIpsLearner(epsilon, max_memories, learn_distance, c=40, d=4, topk=10),
    MemorizedLearner(epsilon, max_memories, learn_distance, c=40, d=4),
    ResidualLearner(epsilon, max_memories, learn_distance, c=40, d=4),
    CorralEnsemble(max_memories, epsilon, eta=.075, fix_count=3000, T=4000, seed=10),
]

if __name__ == '__main__':
    Benchmark.from_file(json).processes(processes).ignore_raise(ignore_raise).evaluate(learners, log).plot_learners()