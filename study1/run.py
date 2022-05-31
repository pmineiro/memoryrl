import os
import math

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from memory import EMT, CMF, DCI
from learners import EpisodicLearner, ComboLearner
from routers import VowpRouter, RandomRouter, EigenRouter
from scorers import RandomScorer, DistScorer, RankScorer, RankScorer2
from tasks import FinalPrintEvaluationTask, SlimOnlineOnPolicyEvalTask
from simulations import LocalSyntheticSimulation, MNIST_LabelFilter, MNIST_SVD
from splitters import LogSplitter, ConstSplitter, NeverSplitter

from coba.contexts     import CobaContext, NullLogger
from coba.environments import Environments, OpenmlSource, ManikSource
from coba.experiments  import Experiment, ClassEnvironmentTask, SimpleEnvironmentTask, OnlineOnPolicyEvalTask
from coba.learners     import VowpalEpsilonLearner, CorralLearner, EpsilonBanditLearner, RandomLearner, VowpalSquarecbLearner

from coba.environments import SimulatedInteraction


log        = None#"./study1/outcomes/full-openml-class-13.log.gz"
config     = {"processes": 8, "chunk_by":'task', 'maxchunksperchild': 0 }

epsilon    = 0.1

if __name__ == '__main__':

   learners = [
      VowpalEpsilonLearner(epsilon, features=["a","xa","xxa"]),
      EpisodicLearner     (epsilon, EMT(6000, EigenRouter(method="RNG"), RankScorer("exp",['x','a','xa']), c=ConstSplitter(100), d=0.00, alpha=0)),
      ComboLearner        (epsilon, EMT(6000, EigenRouter(method="RNG"), RankScorer("exp",['x','a','xa']), c=ConstSplitter(100), d=0.00, alpha=0), "xxa", False, True)
   ]

   environments = Environments.from_template("./study1/experiments/sanity.json", n_shuffle=1, n_take=400)
   #environments = Environments.from_template("./study1/experiments/neurips.json", n_shuffle=50, n_take=4000)
   #environments = Environments.from_template("./study1/experiments/fixed_length.json", n_shuffle=1, n_take=9000)

   result = Experiment(environments, learners, environment_task=SimpleEnvironmentTask(), evaluation_task=FinalPrintEvaluationTask()).config(**config).evaluate(log)
   result.filter_fin().plot_learners(y='reward')
