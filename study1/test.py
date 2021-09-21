import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from simulations import MemorizableSimulation
from learners import ResidualLearner, MemorizedLearner, LoggedCorralLearner, MemorizedVW, CMT_Implemented
from scorers import ClassScorer, AdditionScorer, DistanceScorer, RegrScorer, UCBScorer
from feedbacks import DevFeedback, RwdFeedback
from examplers import PureExampler, DiffSquareExampler

from coba.simulations import ValidationSimulation, OpenmlSimulation
from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner

max_memories = 250
epsilon      = 0.1
d            = 4
c            = 25
megalr       = 0.1

processes  = 6
shuffle    = [1]
take       = 2000

simulations = [
   ValidationSimulation (30000, n_actions=5, action_features=False, make_binary=True),
   ValidationSimulation (30000, n_actions=5, action_features=True , make_binary=True),
   MemorizableSimulation(30000, n_anchors=200, n_actions=2, n_features=5)
]

#BEST
scorer1 = RegrScorer (base="none", exampler=PureExampler(), interactions=["ac","ad","bc","bd","abcd"], ignored=["a","b","c","d"])
scorer2 = ClassScorer(base="none", exampler=PureExampler(), interactions=["ac","ad","bc","bd","abcd"], ignored=["a","b","c","d"])
scorer3 = RegrScorer (base="none", exampler=DiffSquareExampler(), interactions=[], ignored=[])
scorer4 = ClassScorer(base="none", exampler=DiffSquareExampler(), interactions=[], ignored=[])

#SECOND BEST
#scorer1 = RegrScorer (base="l2" , exampler=PureExampler(), interactions=["ac","ad","bc","bd","abcd"], ignored=["a","b","c","d"])
#scorer2 = ClassScorer(base="l2" , exampler=PureExampler(), interactions=["ac","ad","bc","bd","abcd"], ignored=["a","b","c","d"])
#scorer3 = RegrScorer (base="cos", exampler=DiffSquareExampler(), interactions=[], ignored=[])
#scorer4 = ClassScorer(base="l2" , exampler=DiffSquareExampler(), interactions=[], ignored=[])

#[x1,x2,x3] [y1,y2,y3]
#[(x1-y1)**2, (x2-y2)**2, (x3-y3)**2] = ||x-y||^2


#TEST BED
#scorer1 = RegrScorer(base="none",exampler=DiffSquareExampler(), interactions=[], ignored=[])
#scorer2 = RegrScorer(base="mem" ,exampler=DiffSquareExampler(), interactions=[], ignored=[])
#scorer3 = RegrScorer(base="l2"  ,exampler=DiffSquareExampler(), interactions=[], ignored=[])
#scorer4 = ClassScorer(base="cos" ,exampler=DiffSquareExampler(), interactions=[], ignored=[])

cmt_1 = CMT_Implemented(max_memories, scorer=scorer1, signal=RwdFeedback(), c=c, d=d, megalr=megalr)
cmt_2 = CMT_Implemented(max_memories, scorer=scorer2, signal=RwdFeedback(), c=c, d=1, megalr=megalr)
cmt_3 = CMT_Implemented(max_memories, scorer=scorer3, signal=RwdFeedback(), c=c, d=d, megalr=megalr)
cmt_4 = CMT_Implemented(max_memories, scorer=scorer4, signal=RwdFeedback(), c=c, d=1, megalr=megalr)

learners = [
   MemorizedLearner(epsilon, cmt_1),
   MemorizedLearner(epsilon, cmt_2),
   MemorizedLearner(epsilon, cmt_3),
   MemorizedLearner(epsilon, cmt_4),
]

if __name__ == '__main__':
   Benchmark(simulations, take=take, shuffle=shuffle).processes(processes).chunk_by('task').evaluate(learners).plot_learners()