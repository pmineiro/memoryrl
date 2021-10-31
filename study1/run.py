import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from itertools import product

from feedbacks import DeviationFeedback, RewardFeedback
from learners import MemorizedLearner, MemCorralLearner, CMT_Implemented
from routers import Logistic_VW, Logistic_SK
from scorers import RankScorer, RegressionScorer, UCBScorer, Base
from examples import InteractionExample, DifferenceExample

from coba.benchmarks import Benchmark, Result
from coba.learners import VowpalLearner

experiment = 'full6'
processes  = 1
chunk_by   = 'source'

max_memories = 3000
epsilon      = 0.1
d            = 4
c            = 40
megalr       = 0.1

json = f"./study1/experiments/{experiment}.json"
log  = f"./study1/outcomes/{experiment}_1.log.gz"

scorer   = RankScorer(baser=Base("cos") , exampler=DifferenceExample("abs"))
router   = Logistic_VW(power_t=0.0)
feedback = DeviationFeedback("^2")

corral_learners = [
   VowpalLearner("--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --epsilon 0.1 --random_seed 1 --power_t 0.0"), 
   MemorizedLearner(epsilon, CMT_Implemented(6000, scorer=scorer, router=router, feedback=feedback, c=c, d=d, megalr=megalr))
]

learners = [
   MemorizedLearner(epsilon, CMT_Implemented(3000, scorer=scorer, router=router, feedback=feedback, c=c, d=d, megalr=megalr)),
   MemorizedLearner(epsilon, CMT_Implemented(6000, scorer=scorer, router=router, feedback=feedback, c=c, d=d, megalr=megalr)),
   MemCorralLearner(corral_learners, eta=.075, T=10000, type="off-policy"),
   VowpalLearner("--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --epsilon 0.1 --random_seed 1 --power_t 0.0") 
]

if __name__ == '__main__':
   Benchmark.from_file(json).processes(processes).chunk_by(chunk_by).evaluate(learners, log).filter_fin().plot_learners()