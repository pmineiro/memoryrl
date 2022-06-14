import os
import timeit

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from memory import EMT
from learners import EpisodicLearner, ComboLearner
from routers import VowpRouter, RandomRouter, EigenRouter, AbsDevRouter
from scorers import RankScorer
from tasks import FinalPrintEvaluationTask, SlimOnlineOnPolicyEvalTask
from splitters import LogSplitter, ConstSplitter, NeverSplitter
from bounders import NoBounds, LruBounds

from coba.environments import Environments
from coba.experiments  import Experiment, ClassEnvironmentTask, SimpleEnvironmentTask
from coba.learners     import VowpalEpsilonLearner

log    = "./study1/outcomes/neurips-2.log.gz"
config = {"processes": 8, "chunk_by":'task', 'maxchunksperchild': 1 }

epsilon    = 0.1

if __name__ == '__main__':

   learners = [
      VowpalEpsilonLearner(epsilon, features=["a","xa","xxa"]),
      EpisodicLearner     (epsilon, EMT(NoBounds()     , EigenRouter(method="RNG"), RankScorer("exp",['x','a','xa']), splitter=ConstSplitter(100), d=0.00, alpha=0)),
      EpisodicLearner     (epsilon, EMT(NoBounds()     , AbsDevRouter(), RankScorer("exp",['x','a','xa']), splitter=ConstSplitter(100), d=0.00, alpha=0)),
      ComboLearner        (epsilon, EMT(NoBounds()     , EigenRouter(method="RNG"), RankScorer("exp",['x','a','xa']), splitter=ConstSplitter(100), d=0.00, alpha=0), "xxa", False, True),
      ComboLearner        (epsilon, EMT(NoBounds()     , AbsDevRouter(), RankScorer("exp",['x','a','xa']), splitter=ConstSplitter(100), d=0.00, alpha=0), "xxa", False, True),
   ]

   description = "An ablation study comparing projection routing via either Max Var or Max Dev."

   #environments = Environments.from_template("./study1/experiments/sanity.json", n_shuffle=1, n_take=400)
   environments = Environments.from_template("./study1/experiments/neurips.json", n_shuffle=5, n_take=4000)
   #environments = Environments.from_template("./study1/experiments/fixed.json", n_shuffle=1, n_take=10000)

   result = Experiment(environments, learners, description, environment_task=SimpleEnvironmentTask(), evaluation_task=SlimOnlineOnPolicyEvalTask()).config(**config).evaluate(log)
   result.filter_fin().plot_learners(y='reward')
