import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from simulations import MemorizableSimulation
from learners import ResidualLearner, MemorizedLearner, MemCorralLearner, MemorizedVW, CMT_Implemented
from scorers import RankScorer, RegressionScorer, UCBScorer, Base
from feedbacks import DeviationFeedback, RewardFeedback
from examples import InteractionExample, DifferenceExample

from coba.simulations import ValidationSimulation, OpenmlSimulation
from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, CorralLearner

max_memories = 3000
epsilon      = 0.1
d            = 4
c            = 40
megalr       = 0.1

processes  = 4
shuffle    = [1]
take       = 100

simulations = [
   ValidationSimulation(30000)
]

scorers = [
   RankScorer(baser=Base("none"), exampler=DifferenceExample("^2")),
   RankScorer(baser=Base("l1")  , exampler=DifferenceExample("abs")),
   RankScorer(baser=Base("l2^2"), exampler=DifferenceExample("^2")),
   RankScorer(baser=Base("none"), exampler=DifferenceExample("abs")),
   RankScorer(baser=Base("cos") , exampler=DifferenceExample("abs")),
   RankScorer(baser=Base("cos") , exampler=DifferenceExample("abs"))
]

feedbacks = [
   DeviationFeedback("^2"),
   DeviationFeedback("^2"),
   DeviationFeedback("abs"),
   DeviationFeedback("abs"),
   DeviationFeedback("^2"),
   DeviationFeedback("abs")
]

cmts    = [ CMT_Implemented(max_memories, scorer=s, feedback=f, c=c, d=d, megalr=megalr) for s,f in zip(scorers,feedbacks)]
mem_cbs = [ MemorizedLearner(epsilon, cmt) for cmt in cmts]
vw_cb   = VowpalLearner(epsilon=epsilon,seed=1)

learners =  [ MemCorralLearner([vw_cb, mem_cb], eta=.075, T=10000, type="off-policy") for mem_cb in mem_cbs ]
learners += [ vw_cb ]

if __name__ == '__main__':
   Benchmark(simulations, take=take, shuffle=shuffle).processes(processes).chunk_by('task').evaluate(learners).plot_learners()