import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from signals import DevSignal, RwdSignal
from learners import MemorizedLearner, LoggedCorralLearner, CMT_Implemented
from scorers import DiffSubScorer, ClassScorer, DistanceScorer, AdditionScorer, RegrScorer

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner

experiment = 'full'
processes  = 8
chunk_by   = 'source'

max_memories = 3000
epsilon      = 0.1
d            = 4
c            = 40
megalr       = 0.1

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}_12.log"

cmt_1 = CMT_Implemented(max_memories, scorer=ClassScorer(), signal=DevSignal("squared") , c=c, d=d, megalr=megalr)
cmt_2 = CMT_Implemented(max_memories, scorer=ClassScorer(), signal=DevSignal("absolute"), c=c, d=d, megalr=megalr)
cmt_3 = CMT_Implemented(max_memories, scorer=ClassScorer(), signal=RwdSignal()          , c=c, d=d, megalr=megalr)

learners = [
   VowpalLearner(epsilon=epsilon,seed=1),
   MemorizedLearner(epsilon, cmt_3),
   LoggedCorralLearner([VowpalLearner(epsilon=epsilon,seed=1), MemorizedLearner(epsilon, cmt_3)], eta=.075, T=10000, type="off-policy"),
]

if __name__ == '__main__':
   Benchmark.from_file(json).processes(processes).chunk_by(chunk_by).evaluate(learners, log).filter_fin().plot_learners()