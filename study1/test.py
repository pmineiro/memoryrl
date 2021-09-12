import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from simulations import MemorizableSimulation
from learners import ResidualLearner, MemorizedLearner, LoggedCorralLearner, MemorizedVW, CMT_Implemented
from scorers import DiffSubScorer, ClassScorer, AdditionScorer, DistanceScorer, RegrScorer
from signals import DevSignal, RwdSignal

from coba.simulations import ValidationSimulation, OpenmlSimulation
from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner

max_memories = 250
epsilon      = 0.1
d            = 4
c            = 25
megalr       = 0.1

processes  = 8
shuffle    = [1,2]
take       = 1000

simulations = [
   ValidationSimulation (30000, n_actions=5, action_features=False, make_binary=True),
   ValidationSimulation (30000, n_actions=5, action_features=True , make_binary=True),
   MemorizableSimulation(30000, n_anchors=200, n_actions=2, n_features=5)
]

cmt_1 = CMT_Implemented(max_memories, scorer= ClassScorer()            , signal=DevSignal("absolute"), c=c, d=d, megalr=megalr)
cmt_2 = CMT_Implemented(max_memories, scorer= ClassScorer()            , signal=DevSignal("squared") , c=c, d=d, megalr=megalr)
cmt_3 = CMT_Implemented(max_memories, scorer= ClassScorer()            , signal=RwdSignal()          , c=c, d=d, megalr=megalr)
cmt_4 = CMT_Implemented(max_memories, scorer= RegrScorer(base="none")  , signal=RwdSignal()          , c=c, d=d, megalr=megalr)
cmt_5 = CMT_Implemented(max_memories, scorer= RegrScorer(base="dist")  , signal=RwdSignal()          , c=c, d=d, megalr=megalr)
cmt_6 = CMT_Implemented(max_memories, scorer= RegrScorer(base="reward"), signal=RwdSignal()          , c=c, d=d, megalr=megalr)

learners = [
   VowpalLearner(epsilon=epsilon),
   MemorizedLearner(epsilon, cmt_1),
   MemorizedLearner(epsilon, cmt_2),
   MemorizedLearner(epsilon, cmt_3),
   MemorizedLearner(epsilon, cmt_4),
   MemorizedLearner(epsilon, cmt_5),
   MemorizedLearner(epsilon, cmt_6),
]

if __name__ == '__main__':
   Benchmark(simulations, take=take, shuffle=shuffle).processes(processes).chunk_by('task').evaluate(learners).plot_learners()