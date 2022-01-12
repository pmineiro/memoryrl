import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from memory import CMT
from learners import MemorizedLearner
from routers import Logistic_VW, RandomRouter
from scorers import RankScorer, BaseMetric, RegrScorer, TorchScorer
from tasks import RewardLoggingEvaluationTask, FinalPrintEvaluationTask
from examples import DiffExample
from simulations import MemorizableSimulation

from coba.environments import Environments
from coba.experiments import Experiment, ClassEnvironmentTask, SimpleEnvironmentTask, OnlineOnPolicyEvalTask
from coba.learners import VowpalEpsilonLearner, CorralLearner, EpsilonBanditLearner

experiment = 'full6'
json       = f"./study1/experiments/{experiment}.json"
log        = None# f"./study1/outcomes/{experiment}_11.log.gz"
config     = {"processes": 8, "chunk_by":'task' }

max_memories = 6000
epsilon      = 0.1
d            = 4
c            = 14
megalr       = 0.1

rank_cos  = RankScorer(base=BaseMetric("cos"), example=DiffExample("abs"), power_t=0)
rank_exp  = RankScorer(base=BaseMetric("exp"), example=DiffExample("abs"), power_t=0)
regr_cos  = RegrScorer(base=BaseMetric("cos"), example=DiffExample("abs"), power_t=0)
regr_exp  = RegrScorer(base=BaseMetric("exp"), example=DiffExample("abs"), power_t=0)

torch_cos = TorchScorer(base=BaseMetric("cos"), example=DiffExample("abs"), optim=True, v=1)
torch_exp = TorchScorer(base=BaseMetric("exp"), example=DiffExample("abs"), optim=True, v=1)

router = Logistic_VW(power_t=0.0)

# memory_learner = MemorizedLearner(0.1, 'd^2', CMT(max_memories, router, rank_cos , c, d, .25, True), megalr=0, direct_update=False)
vowpal_learner = VowpalEpsilonLearner(epsilon=epsilon, power_t=0)
# corral_learner = CorralLearner([vowpal_learner, memory_learner], eta=.075, T=6000, mode="off-policy")

if __name__ == '__main__':

   learners = [      
      vowpal_learner
   ]

   for e in [0.1,0.3,0.5,0.7,0.9]:
      learners.append(MemorizedLearner(0.1, e, 'd^2', CMT(max_memories, router, rank_cos , c, d, .25, False), explore='e-greedy'))

   #environments = Environments.from_local_synthetic(5000, n_context_features=10, n_actions=2, n_contexts=10).binary().shuffle(range(2))
   environments = Environments.from_linear_synthetic(1000, n_context_features=10, n_actions=2).binary().shuffle(range(2))

   #environments = Environments.from_file(json)

   Experiment(environments, learners, environment_task=ClassEnvironmentTask(), evaluation_task=FinalPrintEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners()
   #result = Experiment(environments, [], environment_task=SimpleEnvironmentTask(), evaluation_task=FinalPrintEvaluationTask()).config(**config).evaluate(log).filter_fin()