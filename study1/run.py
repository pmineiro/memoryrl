import os

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

from coba.environments import Environments
from coba.experiments  import Experiment, ClassEnvironmentTask, SimpleEnvironmentTask, OnlineOnPolicyEvalTask
from coba.learners     import VowpalEpsilonLearner, CorralLearner

experiment = 'full6'
json       = f"./study1/experiments/{experiment}.json"
log        = f"./study1/outcomes/late_experiments3.log.gz"
config     = {"processes": 8, "chunk_by":'task' }

max_memories = 6000
epsilon      = 0.1

router = LogisticRouter(power_t=0.0)

if __name__ == '__main__':

   learners = [
      MemorizedLearner(epsilon, CMT(max_memories, router, RankScorer(power_t=0                     ), c=100, d=2, alpha=.5)),
      MemorizedLearner(epsilon, CMT(max_memories, router, RankScorer(power_t=0, interactions=["xa"]), c=100, d=2, alpha=.5)),
      VowpalEpsilonLearner(epsilon=epsilon, power_t=0)
   ]

   #environments = Environments([LocalSyntheticSimulation(20, n_context_feats=20, n_actions=2, n_contexts=50)]).binary().shuffle(range(1))
   #environments = Environments.from_linear_synthetic(1000, n_context_features=10, n_actions=2).binary().shuffle(range(2))
   #environments = Environments.from_file(json)

   environments = Environments.from_openml(180, take=1000, cat_as_str=False)

   Experiment(environments, learners, environment_task=SimpleEnvironmentTask(), evaluation_task=FinalPrintEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners(span=30)
