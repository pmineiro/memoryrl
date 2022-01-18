from mimetypes import init
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
log        = None#f"./study1/outcomes/full6_22.log.gz"
config     = {"processes": 8, "chunk_by":'task' }

epsilon      = 0.1

if __name__ == '__main__':

   Xs      = [ ["xa", "xxa"] ]
   cs      = [ LogSplitter(26) ]
   ds      = [ .75, 1 ]
   alphas  = [ .25 ]
   init_ws = [ 1 ]
   coins   = [ True ]
   bases   = [ "none", "l2", "cos", "exp" ]

   learners = [
      VowpalEpsilonLearner(epsilon=epsilon, power_t=0)
   ]

   for c, d, alpha, X, w, coin, base in itertools.product(cs,ds,alphas, Xs, init_ws, coins, bases):
      learners.append(MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,X,coin), RankScorer(0,X,w,coin,base), c=c, d=d, alpha=alpha)))

   environments = Environments([LocalSyntheticSimulation(20, n_context_feats=2, n_actions=2, n_contexts=50)]).binary().shuffle(range(1))
   #environments = Environments.from_linear_synthetic(1000, n_context_features=10, n_actions=2).binary().shuffle(range(2))
   #environments = Environments.from_file(json)
   #environments = Environments.from_openml(120, take=6000, cat_as_str=True).scale("min","minmax").shuffle([100])

   Experiment(environments, learners, environment_task=ClassEnvironmentTask(), evaluation_task=FinalPrintEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners()
