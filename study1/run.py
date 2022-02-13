import os
import itertools

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from memory import CMT
from learners import MemorizedLearner
from routers import LogisticRouter, RandomRouter, PCARouter
from scorers import RankScorer, MetricScorer, RankScorer2
from tasks import FinalPrintEvaluationTask
from simulations import LocalSyntheticSimulation, MNIST_LabelFilter, MNIST_SVD
from splitters import LogSplitter, ConstSplitter, NeverSplitter

from coba.environments import Environments
from coba.experiments  import Experiment, ClassEnvironmentTask, SimpleEnvironmentTask, OnlineOnPolicyEvalTask, Result
from coba.learners     import VowpalEpsilonLearner, CorralLearner, EpsilonBanditLearner

experiment = 'full6'
json       = f"./study1/experiments/{experiment}.json"
log        = None#"./study1/outcomes/full6_28.log.gz"
config     = {"processes": 8, "chunk_by":'task' }

epsilon    = 0.1

if __name__ == '__main__':

   learners = [
      #VowpalEpsilonLearner(epsilon, interactions=["xa"]),
      #EpsilonBanditLearner(.1),
      MemorizedLearner(epsilon, CMT(6000, None            , MetricScorer("exp")                   , c=NeverSplitter()   , v=(2,2,1), d=0, alpha=0)),
      MemorizedLearner(epsilon, CMT(6000, PCARouter()     , MetricScorer("exp")                   , c=ConstSplitter(100), v=(2,2,1), d=0, alpha=0)),
      MemorizedLearner(epsilon, CMT(6000, PCARouter()     , RankScorer(0,[],0,"exp",.01,.5,"coin"), c=ConstSplitter(100), v=(2,2,1), d=0, alpha=0)),
      MemorizedLearner(epsilon, CMT(6000, LogisticRouter(), RankScorer(0,[],0,"exp",.01,.5,"coin"), c=ConstSplitter(100), v=(2,2,1), d=0, alpha=0)),
      MemorizedLearner(epsilon, CMT(6000, LogisticRouter(), RankScorer(0,[],0,"exp",.01,.5,"coin"), c=ConstSplitter(100), v=(2,2,1), d=2, alpha=.25))
   ]

   #learners.append(old_learner_2)
   #learners.append(old_learner_3)   
   #learners.append(CorralLearner([VowpalEpsilonLearner(epsilon, interactions=["xa"]), MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,[],True,0), RankScorer(0,[],0,"exp",.01,.5,"coin"), c=cs[0], d=2, alpha=0.25, v=(2,2)))], eta=.075, T=6000, mode="off-policy"))

   environments = Environments([LocalSyntheticSimulation(20, n_context_feats=1, n_actions=2, n_contexts=50)]).binary().shuffle([2]).take(500)
   #environments = Environments.from_openml(554, cat_as_str=True).filter(MNIST_LabelFilter(['9','4'])).filter(MNIST_SVD(30)).scale("min","minmax").shuffle([1]).take(6000)
   #environments = Environments.from_openml(251, cat_as_str=True, take=6000).scale("min","minmax").shuffle([100]).take(300)
   #environments = Environments.from_openml(76, cat_as_str=True, take=6000).scale("min","minmax").shuffle([100]).take(300)
   #environments = Environments.from_openml(73, cat_as_str=True, take=6000).scale("min","minmax").shuffle([100]).take(200)
   #environments = Environments.from_file(json).take(300)

   Experiment(environments, learners, environment_task=SimpleEnvironmentTask(), evaluation_task=FinalPrintEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners(sort="reward")
