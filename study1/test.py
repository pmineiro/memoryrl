import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from simulations import MemorizableSimulation
from learners import ResidualLearner, MemorizedLearner, MemCorralLearner, MemorizedVW, CMT_Implemented
from scorers import ClassScorer, ClassScorer2, AdditionScorer, DistanceScorer, RegrScorer, UCBScorer, Baser
from feedbacks import DevFeedback, RwdFeedback
from examplers import PureExampler, DiffExampler

from coba.simulations import ValidationSimulation, OpenmlSimulation
from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, CorralLearner

max_memories = 3000
epsilon      = 0.1
d            = 4
c            = 40
megalr       = 0.1

processes  = 1
shuffle    = [1]
take       = 10000

simulations = [
   #ValidationSimulation (30000, n_actions=5, action_features=False, make_binary=True),
   #ValidationSimulation (30000, n_actions=5, action_features=True , make_binary=True),
   #MemorizableSimulation(30000, n_anchors=200, n_actions=2, n_features=5)
   OpenmlSimulation(40985, nominal_as_str=True)
]

#BEST
scorer1 = RegrScorer  (exampler=PureExampler())
scorer2 = ClassScorer (exampler=PureExampler())
scorer3 = RegrScorer  (exampler=DiffExampler())
scorer4 = ClassScorer (exampler=DiffExampler())
scorer5 = ClassScorer2()

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

#950-1000 @2000
#750      @2000

cmt_1 = CMT_Implemented(max_memories, router_type='sk', alpha=0.25, scorer=scorer4, feedback=RwdFeedback(), c=c, d=d, megalr=megalr)
cmt_2 = CMT_Implemented(max_memories, router_type='sk', alpha=0.25, scorer=scorer5, feedback=RwdFeedback(), c=c, d=d, megalr=megalr)
cmt_3 = CMT_Implemented(max_memories, router_type='vw', alpha=0.25, scorer=scorer4, feedback=RwdFeedback(), c=c, d=d, megalr=megalr)
#cmt_3 = CMT_Implemented(max_memories, router_type='vw', scorer=scorer3, signal=RwdFeedback(), c=c, d=d, megalr=megalr)
#cmt_4 = CMT_Implemented(max_memories, router_type='vw', scorer=scorer4, signal=RwdFeedback(), c=c, d=d, megalr=megalr)

learners = [
   MemorizedLearner(epsilon, cmt_1),
   #MemorizedLearner(epsilon, cmt_2),
   #MemorizedLearner(epsilon, cmt_3),
   #MemorizedLearner(epsilon, cmt_3),
   #MemorizedLearner(epsilon, cmt_4),
]

if __name__ == '__main__':
   Benchmark(simulations, take=take, shuffle=shuffle).processes(processes).chunk_by('task').evaluate(learners).plot_learners()