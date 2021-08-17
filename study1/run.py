import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import simulations #this import registers our simulations for use in experiments
from learners import ResidualLearner, MemorizedLearner, CorralOffPolicy, MemorizedVW

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, RandomLearner

experiment = 'full'
processes  = 8
chunk_by   = 'source'

max_memories   = 3000
learn_distance = True
epsilon        = 0.1
d              = 4
c              = 40

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}_4.log"

learners = [
   VowpalLearner(epsilon=epsilon),
   MemorizedVW(epsilon),
   MemorizedLearner(epsilon, max_memories, learn_distance, d=d, c=c),
   CorralOffPolicy ([VowpalLearner(epsilon=epsilon,seed=1), MemorizedLearner(epsilon, max_memories, learn_distance, d=d, c=c),], eta=.075, T=10000, seed=1),
   CorralOffPolicy ([VowpalLearner(epsilon=epsilon,seed=1), MemorizedVW(epsilon)], eta=.075, T=10000, seed=1),
   #RandomLearner()
]

if __name__ == '__main__':
   Benchmark.from_file(json).processes(processes).chunk_by(chunk_by).evaluate(learners, log).plot_learners()