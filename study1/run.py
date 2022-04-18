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
from coba.learners     import VowpalEpsilonLearner, CorralLearner, EpsilonBanditLearner, RandomLearner

#experiment = 'full6'
#json       = f"./study1/experiments/{experiment}.json"
#log        = f"./study1/outcomes/{experiment}_31.log.gz"
log        = f"./study1/outcomes/openml-benchmarks.log.gz"
config     = {"processes": 8, "chunk_by":'task' }

epsilon    = 0.1

if __name__ == '__main__':

   learners = [
      RandomLearner(),
      VowpalEpsilonLearner(epsilon, features=["xa"]),
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
   #environments = Environments.from_openml(139, cat_as_str=True, take=6000).scale().shuffle([100]).take(500)
   #environments = Environments.from_file(json)
   #environments  = Environments.from_supervised(ManikSource('.manik/Delicious_data.txt'), take=5000)

   benchmark_tasks = [3,6,11,12,14,15,16,18,22,23,28,29,31,32,37,43,45,49,53,219,2073,2074,2079,3021,3022,3481,3549,3560,3573,3902,3903,3904,3913,3917,3918,3945,7592,7593,9910,9946,9952,9957,9960,9964,9971,9976,9977,9978,9981,9985,10090,10093,10101,14952,14954,14965,14969,14970,34539,125920,125922,146195,146212,146606,146800,146817,146818,146819,146820,146821,146822,146824,146825,167119,167120,167121,167124,167125,167140,167141,167210,168329,168330,168331,168332,168335,168337,168338,168350,168757,168784,168868,168908,168909,168910,168911,168912,189354,189355,189922,190137,190146,190392,190410,190411,190412,211979,211986,233211,233212,233213,233214,233215,317614,359929,359930,359931,359932,359933,359934,359935,359936,359937,359938,359939,359940,359941,359942,359943,359944,359945,359946,359948,359949,359950,359951,359952,359953,359954,359955,359956,359957,359958,359959,359960,359961,359962,359963,359964,359965,359966,359967,359968,359969,359970,359971,359972,359973,359974,359975,359976,359977,359979,359980,359981,359982,359983,359984,359985,359986,359987,359988,359989,359990,359991,359992,359993,359994,360112,360113,360114,360932,360933,360945,360975]
   environments = Environments.from_openml(task_id=benchmark_tasks, cat_as_str=True, take=1000).scale().where(n_interactions=1000)
   Experiment(environments, learners, environment_task=ClassEnvironmentTask(), evaluation_task=FinalPrintEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners(sort="reward")
