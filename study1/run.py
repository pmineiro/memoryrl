import os
import itertools

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from memory import CMT
from learners import MemorizedLearner, MemorizedLearner2
from routers import LogisticRouter, RandomRouter, PCARouter
from scorers import RankScorer, MetricScorer, RankScorer2
from tasks import FinalPrintEvaluationTask
from simulations import LocalSyntheticSimulation, MNIST_LabelFilter, MNIST_SVD
from splitters import LogSplitter, ConstSplitter, NeverSplitter

from coba.environments import Environments, OpenmlSource, ManikSource
from coba.experiments  import Experiment, ClassEnvironmentTask, SimpleEnvironmentTask, OnlineOnPolicyEvalTask, Result
from coba.learners     import VowpalEpsilonLearner, CorralLearner, EpsilonBanditLearner

experiment = 'full6'
json       = f"./study1/experiments/{experiment}.json"
log        = None#f"./study1/outcomes/{experiment}_31.log.gz"
config     = {"processes": 8, "chunk_by":'task' }

epsilon    = 0.1

if __name__ == '__main__':

   learners = [
      MemorizedLearner (epsilon, CMT(6000, PCARouter(), RankScorer("exp"), c=ConstSplitter(100), v=(2,), d=0, alpha=0)),
      MemorizedLearner2(epsilon, CMT(6000, PCARouter(), RankScorer("exp"), c=ConstSplitter(100), v=(2,), d=0, alpha=0)),
   ]

   #learners.append(old_learner_2)
   #learners.append(old_learner_3)
   #learners.append(CorralLearner([VowpalEpsilonLearner(epsilon, interactions=["xa"]), MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,[],True,0), RankScorer(0,[],0,"exp",.01,.5,"coin"), c=cs[0], d=2, alpha=0.25, v=(2,2)))], eta=.075, T=6000, mode="off-policy"))
   #environments = Environments([LocalSyntheticSimulation(20, n_context_feats=5, n_actions=2, n_contexts=50)]).binary().shuffle([2,3])
   #environments = Environments.from_openml(554, cat_as_str=True).filter(MNIST_LabelFilter(['9','4'])).filter(MNIST_SVD(30)).shuffle([1]).take(200)
   #environments = Environments.from_openml(251, cat_as_str=True, take=6000).scale().shuffle([100]).take(300)
   #environments = Environments.from_openml(76, cat_as_str=True, take=6000).scale().shuffle([100]).take(300)
   #environments = Environments.from_openml(180, cat_as_str=True, take=6000).scale().shuffle([100]).take(1000)
   #environments = Environments.from_openml(722, cat_as_str=True, take=6000).scale().shuffle([100]).take(500)
   environments = Environments.from_file(json)
   #environments  = Environments.from_supervised(ManikSource('.manik/Delicious_data.txt'), take=5000)

   Experiment(environments, learners, environment_task=SimpleEnvironmentTask(), evaluation_task=FinalPrintEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners(sort="reward")
