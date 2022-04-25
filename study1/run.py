import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from memory import CMT
from learners import MemorizedLearner1, MemorizedLearner2
from routers import LogRouter, RandomRouter, PCARouter
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
log        = "./study1/outcomes/full-openml-class-2.log.gz"
config     = {"processes": 128, "chunk_by":'source' }

epsilon    = 0.1

if __name__ == '__main__':

   learners = [
      RandomLearner(),
      EpsilonBanditLearner(epsilon),
      VowpalEpsilonLearner(epsilon, features=["a","xa"]),
      VowpalEpsilonLearner(epsilon, features=["a","xa","xxa"]),
      MemorizedLearner1(epsilon, CMT(6000, PCARouter(['x']), RankScorer("exp",['x','a','xa'],False), c=ConstSplitter(100), v=(2,), d=0, alpha=0)       ),
      MemorizedLearner1(epsilon, CMT(6000, PCARouter(['x']), RankScorer("exp",['x','a','xa'],True ), c=ConstSplitter(100), v=(2,), d=0, alpha=0)       ),
      MemorizedLearner2(epsilon, CMT(6000, PCARouter(['x']), RankScorer("exp",['x','a','xa'],True ), c=ConstSplitter(100), v=(2,), d=0, alpha=0), "xa" ),
      MemorizedLearner2(epsilon, CMT(6000, PCARouter(['x']), RankScorer("exp",['x','a','xa'],True ), c=ConstSplitter(100), v=(2,), d=0, alpha=0), "xxa"),
      #MemorizedLearner1(epsilon, CMT(6000, PCARouter(['xa']), RankScorer("exp",['x','a','xa'],False), c=ConstSplitter(100), v=(2,), d=0, alpha=0)),
      #MemorizedLearner2(epsilon, CMT(6000, PCARouter(['xa']), RankScorer("exp",['x','a','xa'],False), c=ConstSplitter(100), v=(2,), d=0, alpha=0)),
      #MemorizedLearner2(epsilon, CMT(6000, PCARouter(['xa']), RankScorer("exp",['xa'],False), c=ConstSplitter(100), v=(2,), d=0, alpha=0)),
      #MemorizedLearner2(epsilon, CMT(6000, PCARouter(['xa']), RankScorer("exp",['xa'],True ), c=ConstSplitter(100), v=(2,), d=0, alpha=0)),
      #MemorizedLearner1(epsilon, CMT(6000, PCARouter(['xa']), RankScorer("exp"), c=ConstSplitter(100), v=(2,), d=0, alpha=0)),
      #MemorizedLearner2(epsilon, CMT(6000, PCARouter(['xa']), RankScorer("exp"), c=ConstSplitter(100), v=(2,), d=0, alpha=0)),
      #MemorizedLearner2(epsilon, CMT(6000, PCARouter(['xa']), RankScorer("exp",['x','a','xa'],False), c=ConstSplitter(100), v=(2,), d=0, alpha=0)),
      #MemorizedLearner2(epsilon, CMT(6000, PCARouter(['xa']), RankScorer("exp",['x','a','xa'],True ), c=ConstSplitter(100), v=(2,), d=0, alpha=0)),
   ]

   #learners.append(old_learner_2)
   #learners.append(old_learner_3)
   #learners.append(CorralLearner([VowpalEpsilonLearner(epsilon, interactions=["xa"]), MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,[],True,0), RankScorer(0,[],0,"exp",.01,.5,"coin"), c=cs[0], d=2, alpha=0.25, v=(2,2)))], eta=.075, T=6000, mode="off-policy"))
   #environments = Environments([LocalSyntheticSimulation(20, n_context_feats=5, n_actions=2, n_contexts=50)]).binary().shuffle([2,3])
   #environments = Environments.from_openml(554, cat_as_str=True).filter(MNIST_LabelFilter(['9','4'])).filter(MNIST_SVD(30)).scale().shuffle([1]).take(1000)
   #environments = Environments.from_openml(251, cat_as_str=True, take=6000).scale().shuffle([100]).take(300)
   #environments = Environments.from_openml(76, cat_as_str=True, take=6000).scale().shuffle([100]).take(300)
   #environments = Environments.from_openml(180, cat_as_str=True, take=6000).scale().shuffle([100]).take(500)
   #environments = Environments.from_openml(722, cat_as_str=True, take=6000).scale().shuffle([100]).take(500)
   #environments = Environments.from_openml(139, cat_as_str=True, take=6000).scale().shuffle([100]).take(500)
   #environments = Environments.from_file(json)
   #environments  = Environments.from_supervised(ManikSource('.manik/Delicious_data.txt'), take=5000)

   #environments = Environments.from_openml(554, cat_as_str=True).filter(MNIST_LabelFilter(['9','4'])).filter(MNIST_SVD(30)) + Environments.from_openml([251,76,180,722,139], cat_as_str=True, take=6000)
   #environments = environments.scale().shuffle([100]).take(1000)

   benchmark_tasks = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 24, 26, 28, 29, 31, 32, 37, 43, 45, 49, 53, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 229, 230, 334, 2071, 2072, 2073, 2074, 2076, 2079, 2135, 2137, 2138, 2139, 2140, 2142, 2144, 2146, 2148, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2266, 2267, 2270, 2276, 3017, 3020, 3021, 3022, 3048, 3481, 3504, 3505, 3506, 3510, 3549, 3560, 3573, 3588, 3591, 3593, 3600, 3601, 3618, 3668, 3672, 3684, 3686, 3688, 3698, 3708, 3711, 3712, 3745, 3764, 3786, 3822, 3840, 3844, 3882, 3884, 3893, 3897, 3899, 3902, 3903, 3904, 3913, 3917, 3918, 3944, 3945, 3947, 3950, 3953, 3954, 7290, 7291, 7293, 7295, 7592, 7593, 7608, 9890, 9891, 9892, 9910, 9911, 9920, 9930, 9943, 9945, 9946, 9952, 9957, 9959, 9960, 9964, 9965, 9966, 9971, 9972, 9974, 9976, 9977, 9978, 9981, 9983, 9985, 9986, 10090, 10091, 10093, 10101, 10106, 14952, 14953, 14954, 14963, 14965, 14969, 14970, 34539, 52950, 75127, 75144, 125920, 125922, 145827, 146086, 146089, 146090, 146091, 146093, 146094, 146095, 146096, 146133, 146195, 146199, 146212, 146606, 146688, 146800, 146806, 146809, 146817, 146818, 146819, 146820, 146821, 146822, 146824, 146825, 167039, 167042, 167043, 167044, 167045, 167050, 167083, 167112, 167119, 167120, 167121, 167124, 167125, 167130, 167140, 167141, 168294, 168298, 168329, 168330, 168331, 168332, 168335, 168337, 168338, 168339, 168350, 168739, 168757, 168784, 168868, 168882, 168908, 168909, 168910, 168911, 168912, 189354, 189355, 189773, 189783, 189785, 189877, 189922, 190137, 190146, 190392, 190410, 190411, 190412, 190424, 211687, 211715, 211717, 211719, 211979, 211983, 211986, 317599, 317601, 359953, 359954, 359955, 359956, 359957, 359958, 359959, 359960, 359961, 359962, 359963, 359964, 359965, 359966, 359967, 359968, 359969, 359970, 359971, 359972, 359973, 359974, 359975, 359976, 359977, 359979, 359980, 359981, 359982, 359983, 359984, 359985, 359986, 359987, 359988, 359989, 359990, 359991, 359992, 359993, 359994, 360112, 360113, 360114, 360975]
   environments = Environments.from_openml(task_id=benchmark_tasks, cat_as_str=True, take=3000).scale()
   
   Experiment(environments, learners, environment_task=SimpleEnvironmentTask(), evaluation_task=OnlineOnPolicyEvalTask()).config(**config).evaluate(log).filter_fin().plot_learners(sort="y")
