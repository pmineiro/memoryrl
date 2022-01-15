import os
import itertools

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from memory import CMT
from learners import MemorizedLearner
from routers import LogisticRouter
from scorers import RankScorer
from tasks import FinalPrintEvaluationTask
from simulations import LocalSyntheticSimulation
from splitters import LogSplitter, ConstSplitter

from coba.environments import Environments
from coba.experiments  import Experiment, ClassEnvironmentTask, SimpleEnvironmentTask, OnlineOnPolicyEvalTask
from coba.learners     import VowpalEpsilonLearner, CorralLearner

experiment = 'full6'
json       = f"./study1/experiments/{experiment}.json"
log        = f"./study1/outcomes/full6_15.log.gz"
config     = {"processes": 1, "chunk_by":'task' }

epsilon      = 0.1

if __name__ == '__main__':

   Xs     = [ [], ["xa"], ["xa","xxa"] ]
   cs     = [ ConstSplitter(100), ConstSplitter(120), LogSplitter(17), LogSplitter(20) ]
   ds     = [ 1, 2 ]
   alphas = [ .5, .75 ]

   learners = [
      VowpalEpsilonLearner(epsilon=epsilon, power_t=0)
   ]

   for c, d, alpha, X in itertools.product(cs,ds,alphas, Xs):
      learners.append(MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,X), RankScorer(0,X), c=c, d=d, alpha=alpha)))

   #environments = Environments([LocalSyntheticSimulation(20, n_context_feats=3, n_actions=2, n_contexts=50)]).binary().shuffle(range(3))
   #environments = Environments.from_linear_synthetic(1000, n_context_features=10, n_actions=2).binary().shuffle(range(2))
   environments = Environments.from_file(json)

   Experiment(environments, learners, environment_task=ClassEnvironmentTask(), evaluation_task=FinalPrintEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners()
