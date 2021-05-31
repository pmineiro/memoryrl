import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import simulations #this import registers our simulations for use in experiments
from learners import ResidualLearner, MemorizedLearner, CorralEnsemble, CorralRejection, FullFeedbackMemLearner, FullFeedbackVowpalLearner

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, UcbBanditLearner

experiment       = "large"
processes        = 8
seed             = 10
ignore_raise     = False

max_memories   = 2000
learn_distance = True
epsilon        = 0.1

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}.log"

learners = [
    FullFeedbackMemLearner(epsilon , max_memories, learn_distance),
    FullFeedbackVowpalLearner(epsilon),
    VowpalLearner(epsilon=epsilon, seed=seed),
    UcbBanditLearner()
]

if __name__ == '__main__':
    Benchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).ignore_raise(ignore_raise).evaluate(learners, log, seed).plot_learners()