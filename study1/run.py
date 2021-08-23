import os

from coba.simulations.core import ValidationSimulation

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import simulations #this import registers our simulations for use in experiments
from learners import ResidualLearner, MemorizedLearner, LoggedCorralLearner, MemorizedVW

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, RandomLearner

experiment = 'madish'
processes  = 10
chunk_by   = 'source'

max_memories   = 3000
learn_distance = True
epsilon        = 0.1
d              = 4
c              = 20
megalr         = 0.1
scorer         = 'vw'

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}_9.log"

learners = [
   VowpalLearner(epsilon=epsilon),
   MemorizedVW(epsilon),
   MemorizedLearner(epsilon, max_memories, learn_distance, scorer=scorer, d=d, c=c, megalr=megalr),
   ResidualLearner(epsilon, max_memories, learn_distance, scorer=scorer, d=d, c=c, megalr=megalr),
   LoggedCorralLearner ([VowpalLearner(epsilon=epsilon,seed=1), MemorizedLearner(epsilon, max_memories, learn_distance, scorer=scorer, d=d, c=c, megalr=megalr)], eta=.075, T=10000, type="off-policy", seed=1),
   LoggedCorralLearner ([VowpalLearner(epsilon=epsilon,seed=1), MemorizedVW(epsilon)], eta=.075, T=10000, type="off-policy", seed=1),
   LoggedCorralLearner ([VowpalLearner(epsilon=epsilon,seed=1), MemorizedLearner(epsilon, max_memories, learn_distance, scorer=scorer, d=d, c=c, megalr=megalr)], eta=.075, T=10000, type="rejection", seed=1),
   #RandomLearner()
]

if __name__ == '__main__':
   Benchmark.from_file(json).processes(processes).chunk_by(chunk_by).evaluate(learners, log).plot_learners()