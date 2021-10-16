import os

from coba.learners.core import RandomLearner

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from simulations import MemorizableSimulation
from learners import ResidualLearner, MemorizedLearner, MemCorralLearner, CMT_Implemented
from scorers import RankScorer, RegressionScorer, UCBScorer, Base
from feedbacks import DeviationFeedback, RewardFeedback
from examples import InteractionExample, DifferenceExample, OG_DifferenceExample

from coba.simulations import ValidationSimulation, OpenmlSimulation
from coba.benchmarks import Benchmark
from coba.learners import VowpalLearner, CorralLearner

max_memories = 3000
epsilon      = 0.1
d            = 4
c            = 40
megalr       = 0.1

processes  = 4
shuffle    = [1,2]
take       = 1000

simulations = [
   ValidationSimulation(500,sparse=False)
]

scorers = [
   RankScorer(baser=Base("cos") , exampler=DifferenceExample("abs"))
]

feedbacks = [
   DeviationFeedback("^2"),
]

vw_cb = VowpalLearner(epsilon=epsilon,seed=1,power_t=0)

cmts     = [ CMT_Implemented(max_memories, scorer=s, feedback=f, c=c, d=d, megalr=megalr) for s,f in zip(scorers,feedbacks)]
mem_cbs  = [ MemorizedLearner(epsilon, cmt) for cmt in cmts]
learners = [ MemCorralLearner([vw_cb, mem_cb], eta=.075, T=10000, type="off-policy") for mem_cb in mem_cbs ]

learners += [ vw_cb ]

if __name__ == '__main__':
   Benchmark(simulations, take=take, shuffle=shuffle).processes(processes).chunk_by('task').evaluate(learners).plot_learners()