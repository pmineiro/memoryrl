import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

import simulations

from feedbacks import DevFeedback, RwdFeedback
from learners import MemorizedLearner, MemCorralLearner, CMT_Implemented
from scorers import ClassScorer, ClassScorer2, DistanceScorer, AdditionScorer, RegrScorer, UCBScorer, Baser
from examplers import PureExampler, DiffExampler

from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner

experiment = 'full'
processes  = 1
chunk_by   = 'source'

max_memories = 3000
epsilon      = 0.1
d            = 4
c            = 40
megalr       = 0.1

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}_17.log"

vw_score_types = [ ClassScorer ]

vw_examplers = [
   DiffExampler("square"),
   DiffExampler("abs")
]

vw_basers = [
   Baser("none"),
   Baser("l1"),
   Baser("l2"),
   Baser("sql2"),
   Baser("cos")
]

vw_scorers_instantiated = [ t(baser=b, exampler=e) for t in vw_score_types for b in vw_basers for e in vw_examplers ]
sk_scorers_instantiated = [ ClassScorer2() ]

scorers   = vw_scorers_instantiated + sk_scorers_instantiated
feedbacks = [ RwdFeedback(), DevFeedback("squared"), DevFeedback("absolute") ]

cmts      = [ CMT_Implemented(max_memories, scorer=s, feedback=f, c=c, d=d, megalr=megalr) for s in scorers for f in feedbacks ]
cmts     += [ CMT_Implemented(max_memories, scorer=ClassScorer(), feedback=RwdFeedback(), c=c, d=d, megalr=megalr, sort=True)]

mem_cbs   = [ MemorizedLearner(epsilon, cmt) for cmt in cmts]
vw_cb     = VowpalLearner(epsilon=epsilon,seed=1)

learners  = [ MemCorralLearner([vw_cb, mem_cb], eta=.075, T=10000, type="off-policy") for mem_cb in mem_cbs ]

if __name__ == '__main__':
   Benchmark.from_file(json).processes(processes).chunk_by(chunk_by).evaluate(learners, log).filter_fin().plot_learners()