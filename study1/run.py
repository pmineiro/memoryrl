import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from feedbacks import DevFeedback, RwdFeedback
from learners import MemorizedLearner, LoggedCorralLearner, CMT_Implemented
from scorers import ClassScorer, DistanceScorer, AdditionScorer, RegrScorer, UCBScorer
from examplers import PureExampler, DiffSquareExampler

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner

experiment = 'full'
processes  = 6
chunk_by   = 'task'

max_memories = 3000
epsilon      = 0.1
d            = 4
c            = 40
megalr       = 0.1

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}_16.log"

scorer1 = RegrScorer (base="none", exampler=PureExampler(), interactions=["ac","ad","bc","bd","abcd"], ignored=["a","b","c","d"])
scorer2 = ClassScorer(base="none", exampler=PureExampler(), interactions=["ac","ad","bc","bd","abcd"], ignored=["a","b","c","d"])
scorer3 = RegrScorer (base="none", exampler=DiffSquareExampler(), interactions=[], ignored=[])
scorer4 = ClassScorer(base="none", exampler=DiffSquareExampler(), interactions=[], ignored=[])
scorer5 = RegrScorer (base="cos" , exampler=DiffSquareExampler(), interactions=[], ignored=[])
scorer6 = ClassScorer(base="cos" , exampler=DiffSquareExampler(), interactions=[], ignored=[])

cmt_1 = CMT_Implemented(max_memories, scorer=scorer1, signal=RwdFeedback(), c=c, d=d, megalr=megalr)
cmt_2 = CMT_Implemented(max_memories, scorer=scorer2, signal=RwdFeedback(), c=c, d=d, megalr=megalr)
cmt_3 = CMT_Implemented(max_memories, scorer=scorer3, signal=RwdFeedback(), c=c, d=d, megalr=megalr)
cmt_4 = CMT_Implemented(max_memories, scorer=scorer4, signal=RwdFeedback(), c=c, d=d, megalr=megalr)
cmt_5 = CMT_Implemented(max_memories, scorer=scorer5, signal=RwdFeedback(), c=c, d=d, megalr=megalr)
cmt_6 = CMT_Implemented(max_memories, scorer=scorer6, signal=RwdFeedback(), c=c, d=d, megalr=megalr)

learners = [
   LoggedCorralLearner([VowpalLearner(epsilon=epsilon,seed=1), MemorizedLearner(epsilon, cmt_1)], eta=.075, T=10000, type="off-policy"),
   LoggedCorralLearner([VowpalLearner(epsilon=epsilon,seed=1), MemorizedLearner(epsilon, cmt_2)], eta=.075, T=10000, type="off-policy"),
   LoggedCorralLearner([VowpalLearner(epsilon=epsilon,seed=1), MemorizedLearner(epsilon, cmt_3)], eta=.075, T=10000, type="off-policy"),
   LoggedCorralLearner([VowpalLearner(epsilon=epsilon,seed=1), MemorizedLearner(epsilon, cmt_4)], eta=.075, T=10000, type="off-policy"),
   LoggedCorralLearner([VowpalLearner(epsilon=epsilon,seed=1), MemorizedLearner(epsilon, cmt_5)], eta=.075, T=10000, type="off-policy"),
   LoggedCorralLearner([VowpalLearner(epsilon=epsilon,seed=1), MemorizedLearner(epsilon, cmt_6)], eta=.075, T=10000, type="off-policy"),
]

if __name__ == '__main__':
   Benchmark.from_file(json).processes(processes).chunk_by(chunk_by).evaluate(learners, log).filter_fin().plot_learners()