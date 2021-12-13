import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from memory import CMT
from learners import OmegaDiffLearner, RewarDirectLearner
from routers import Logistic_VW, RandomRouter
from scorers import RankScorer, BaseMetric, RegrScorer
from tasks import RewardLoggingEvaluationTask
from examples import DiffExample
from simulations import MemorizableSimulation

from coba.environments import Environments
from coba.experiments import Experiment, ClassEnvironmentTask, SimpleEnvironmentTask, OnPolicyEvaluationTask
from coba.learners import VowpalLearner, CorralLearner, EpsilonBanditLearner

experiment = 'full6'
json       = f"./study1/experiments/{experiment}.json"
log        = None#f"./study1/outcomes/{experiment}_8.log.gz"
config     = {"processes":8, "chunk_by":'task' }

max_memories = 6000
epsilon      = 0.1
d            = 4
c            = 15
megalr       = 0.1

rank_cos = RankScorer(base=BaseMetric("cos"), example=DiffExample("abs"), power_t=0)
rank_exp = RankScorer(base=BaseMetric("exp"), example=DiffExample("abs"), power_t=0)
regr_cos = RegrScorer(base=BaseMetric("cos"), example=DiffExample("abs"), power_t=0)
regr_exp = RegrScorer(base=BaseMetric("exp"), example=DiffExample("abs"), power_t=0)

router = Logistic_VW(power_t=0.0)

omega_learner  = OmegaDiffLearner(epsilon, CMT(max_memories, router, rank_cos, c, d), signal='^2', megalr=0.1)
vowpal_learner = VowpalLearner(epsilon=epsilon, power_t=0)
corral_learner = CorralLearner([vowpal_learner, omega_learner], eta=.075, T=10000, type="off-policy")

if __name__ == '__main__':
   
   learners = []

   for megalr in [0,0.1]:
      learners.extend([
         OmegaDiffLearner  (epsilon, CMT(max_memories, router, rank_cos, c, d),signal ='^2'  , megalr=megalr),
         OmegaDiffLearner  (epsilon, CMT(max_memories, router, rank_exp, c, d),signal ='^2'  , megalr=megalr),
         OmegaDiffLearner  (epsilon, CMT(max_memories, router, rank_cos, c, d),signal ='^2'  , megalr=megalr, sort=True),
         OmegaDiffLearner  (epsilon, CMT(max_memories, router, rank_exp, c, d),signal ='^2'  , megalr=megalr, sort=True),
         RewarDirectLearner(epsilon, CMT(max_memories, router, rank_cos, c, d),explore="each", megalr=megalr),
         RewarDirectLearner(epsilon, CMT(max_memories, router, rank_exp, c, d),explore="each", megalr=megalr),
         RewarDirectLearner(epsilon, CMT(max_memories, router, regr_cos, c, d),explore="each", megalr=megalr),
         RewarDirectLearner(epsilon, CMT(max_memories, router, regr_exp, c, d),explore="each", megalr=megalr),
      ])

   environments = [
      MemorizableSimulation(n_interactions=4000,n_features=2,n_actions=2,n_context=300),
      MemorizableSimulation(n_interactions=4000,n_features=2,n_actions=4,n_context=300),
      #MemorizableSimulation(n_interactions=1500,n_features=2,n_actions=8,n_context=10)
   ]

   #environments = Environments.from_file(json)

   #shuffle_env = Environments(environments)
   #Experiment(environments, learners, environment_task=ClassEnvironmentTask(), evaluation_task=RewardLoggingEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners()
   Experiment(environments, learners, environment_task=SimpleEnvironmentTask(), evaluation_task=OnPolicyEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners()