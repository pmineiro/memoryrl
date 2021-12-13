import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from memory import CMT
from learners import OmegaDiffLearner
from routers import Logistic_VW, RandomRouter
from scorers import RankScorer, BaseMetric, RandomScorer
from tasks import RewardLoggingEvaluationTask
from examples import DiffExample
from simulations import MemorizableSimulation

from coba.environments import Environments
from coba.experiments import Experiment, ClassEnvironmentTask, SimpleEnvironmentTask, OnPolicyEvaluationTask
from coba.learners import VowpalLearner, CorralLearner, EpsilonBanditLearner

experiment = 'full6'
json       = f"./study1/experiments/{experiment}.json"
log        = None#f"./study1/outcomes/{experiment}_6.log.gz"
config     = {"processes":8, "chunk_by":'source' }

max_memories = 6000
epsilon      = 0.1
d            = 4
c            = 40
megalr       = 0.1

scorer   = RankScorer(base=BaseMetric("exp"), example=DiffExample("abs"), power_t=0)
router   = Logistic_VW(power_t=0.0)

memorized_learner = OmegaDiffLearner(epsilon, '^2', CMT(max_memories, router, scorer, c, d), megalr=0.1)
vowpal_learner    = VowpalLearner(epsilon=epsilon, power_t=0)
corral_learner    = CorralLearner([vowpal_learner, memorized_learner], eta=.075, T=10000, type="off-policy")

if __name__ == '__main__':
   learners = [
      memorized_learner,
      vowpal_learner
   ]

   base_env = []

   environments = [
      MemorizableSimulation(n_interactions=4000,n_features=2,n_actions=2,n_context=20),
      #MemorizableSimulation(n_interactions=1500,n_features=2,n_actions=4,n_context=10),
      #MemorizableSimulation(n_interactions=1500,n_features=2,n_actions=8,n_context=10)
   ]
   
   #environments = Environments.from_file(json)

   #shuffle_env = Environments(environments)
   Experiment(environments, learners, environment_task=ClassEnvironmentTask(), evaluation_task=RewardLoggingEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners(xlim=(0,4000), each=True)
   #Experiment(environments, learners, environment_task=SimpleEnvironmentTask(), evaluation_task=OnPolicyEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners()