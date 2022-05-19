import os

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from memory import CMT, CMF, DCI
from learners import MemorizedLearner1, MemorizedLearner2
from routers import VowpRouter, RandomRouter, ProjRouter
from scorers import RandomScorer, DistScorer, RankScorer, RankScorer2
from tasks import FinalPrintEvaluationTask, SlimOnlineOnPolicyEvalTask
from simulations import LocalSyntheticSimulation, MNIST_LabelFilter, MNIST_SVD
from splitters import LogSplitter, ConstSplitter, NeverSplitter

from coba.environments import Environments, OpenmlSource, ManikSource
from coba.experiments  import Experiment, ClassEnvironmentTask, SimpleEnvironmentTask, OnlineOnPolicyEvalTask
from coba.learners     import VowpalEpsilonLearner, CorralLearner, EpsilonBanditLearner, RandomLearner, VowpalSquarecbLearner

#experiment = 'full6'
#json       = f"./study1/experiments/{experiment}.json"
#log        = f"./study1/outcomes/{experiment}_31.log.gz"
log        = "./study1/outcomes/full-openml-class-13.log.gz"
config     = {"processes": 128, "chunk_by":'task', 'maxchunksperchild': 2 }

epsilon    = 0.1

if __name__ == '__main__':

   learners = [
      #VowpalEpsilonLearner(epsilon, features=["a","xa","xxa"]),

      #MemorizedLearner1(epsilon, CMF(5, DistScorer("exp",['x','a','xa']), CMT( 10000, ProjRouter(['x'],"RNG",samples=90), RandomScorer()                  , c=ConstSplitter(100), v=(2,), d=0.00, alpha=0, rng=None))                    ),
      #MemorizedLearner1(epsilon, CMF(5, RankScorer("exp",['x','a','xa']), CMT( 10000, ProjRouter(['x'],"RNG",samples=90), RandomScorer()                  , c=ConstSplitter(100), v=(2,), d=0.00, alpha=0, rng=None))                    ),
      #MemorizedLearner2(epsilon, CMF(5, DistScorer("exp",['x','a','xa']), CMT( 10000, ProjRouter(['x'],"RNG",samples=90), RandomScorer()                  , c=ConstSplitter(100), v=(2,), d=0.00, alpha=0, rng=None)), "xxa", False, True),
      #MemorizedLearner2(epsilon, CMF(5, RankScorer("exp",['x','a','xa']), CMT( 10000, ProjRouter(['x'],"RNG",samples=90), RandomScorer()                  , c=ConstSplitter(100), v=(2,), d=0.00, alpha=0, rng=None)), "xxa", False, True),

      #is regression learner better?
         #NO

      #does abs lose information?
         #NO

      #remove coin and use init_arg 1?
         #NO

      #MemorizedLearner1(epsilon, CMT(6000, None, DistScorer ("exp",['x','a','xa']), c=NeverSplitter(), d=0.00, alpha=0)),
      #MemorizedLearner1(epsilon, CMT(6000, None, RankScorer ("exp",['x','a','xa']), c=NeverSplitter(), d=0.00, alpha=0)),

      MemorizedLearner1(epsilon, CMT(6000, VowpRouter(base=None ,fixed=False), RankScorer ("exp",['x','a','xa']), c=ConstSplitter(100), d=1.00, alpha=0.25)),
      MemorizedLearner1(epsilon, CMT(6000, VowpRouter(base="RNG",fixed=False), RankScorer ("exp",['x','a','xa']), c=ConstSplitter(100), d=1.00, alpha=0.25)),
      MemorizedLearner1(epsilon, CMT(6000, VowpRouter(base=None ,fixed=True ), RankScorer ("exp",['x','a','xa']), c=ConstSplitter(100), d=0.00, alpha=1.00)),
      MemorizedLearner1(epsilon, CMT(6000, VowpRouter(base="RNG",fixed=True ), RankScorer ("exp",['x','a','xa']), c=ConstSplitter(100), d=0.00, alpha=1.00)),
      MemorizedLearner1(epsilon, CMT(6000, ProjRouter(proj="RNG"            ), RankScorer ("exp",['x','a','xa']), c=ConstSplitter(100), d=0.00, alpha=0)),
      MemorizedLearner1(epsilon, CMT(6000, ProjRouter(proj="RNG"            ), RankScorer2("exp",['x','a'     ]), c=ConstSplitter(100), d=0.00, alpha=0)),

      #MemorizedLearner1(epsilon, CMT(6000, None, RankScorer("l2",['x','a','xa']), c=NeverSplitter(), v=(2,), d=0.00, alpha=0)),

      #MemorizedLearner1(epsilon, CMT(6000, None, RankScorer("l2",['x','a','xa']), c=NeverSplitter(), v=(2,), d=0.00, alpha=0)),
      #MemorizedLearner1(epsilon, CMT(6000, None, RankScorer("l1",['x','a','xa']), c=NeverSplitter(), v=(2,), d=0.00, alpha=0)),
      #MemorizedLearner1(epsilon, CMF(2,  6000, ProjRouter(['x'],"PCA",samples=90), RankScorer("exp",['x','a','xa']), c=ConstSplitter(100), v=(2,), d=0.00, alpha=0)),
      #MemorizedLearner1(epsilon, CMF(6,  6000, ProjRouter(['x'],"RNG",samples=90), RankScorer("exp",['x','a','xa']), c=ConstSplitter(100), v=(2,), d=0.00, alpha=0)),
      #MemorizedLearner1(epsilon, CMF(20, 6000, ProjRouter(['x'],"RNG",samples=90), DistScorer("exp",['x','a','xa']), c=ConstSplitter(20 ), v=(2,), d=0.00, alpha=0, max_depth=4)),
      #MemorizedLearner1(epsilon, CMT(    6000, ProjRouter(['x'],"PCA",samples=90), RankScorer("exp",['x','a','xa']), c=ConstSplitter(100), v=(2,), d=0.00, alpha=0)),
      #MemorizedLearner1(epsilon, DCI(100, 30, 100, "RNG", RankScorer("exp",['x','a','xa']))),
   ]

   #learners.append(old_learner_2)
   #learners.append(old_learner_3)
   #learners.append(CorralLearner([VowpalEpsilonLearner(epsilon, interactions=["xa"]), MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,[],True,0), RankScorer(0,[],0,"exp",.01,.5,"coin"), c=cs[0], d=2, alpha=0.25, v=(2,2)))], eta=.075, T=6000, mode="off-policy"))
   #environments = Environments([LocalSyntheticSimulation(20, n_context_feats=5, n_actions=2, n_contexts=50)]).binary().shuffle([2,3])
   #environments = Environments.from_openml(554, cat_as_str=True).filter(MNIST_LabelFilter(['9','4'])).filter(MNIST_SVD(30)).scale().shuffle([100]).take(1000)
   #environments = Environments.from_openml(251, cat_as_str=True, take=6000).scale().shuffle([100]).take(100)
   #environments = Environments.from_openml(76, cat_as_str=True, take=6000).scale().shuffle([100]).take(100)
   #environments = Environments.from_openml(180, cat_as_str=True, take=6000).scale().shuffle([100]).take(20)
   #environments = Environments.from_openml(722, cat_as_str=True, take=6000).scale().shuffle([100]).take(20)
   #environments = Environments.from_openml(139, cat_as_str=True, take=6000).scale().shuffle([100]).take(200)
   #environments = Environments.from_file(json)
   #environments  = Environments.from_supervised(ManikSource('.manik/Delicious_data.txt'), take=5000)

   #environments = Environments.from_openml(554, cat_as_str=True).filter(MNIST_LabelFilter(['9','4'])).filter(MNIST_SVD(30)) + Environments.from_openml([251,76,180,722,139], cat_as_str=True, take=6000)
   #environments = environments.scale().shuffle([1]).take(500)

   benchmark_tasks = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 24, 26, 28, 29, 31, 32, 37, 43, 45, 49, 53, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 229, 230, 334, 2071, 2072, 2073, 2074, 2076, 2079, 2135, 2137, 2138, 2139, 2140, 2142, 2144, 2146, 2148, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2266, 2267, 2270, 2276, 3017, 3020, 3021, 3022, 3048, 3481, 3504, 3505, 3506, 3510, 3549, 3560, 3573, 3588, 3591, 3593, 3600, 3601, 3618, 3668, 3672, 3684, 3686, 3688, 3698, 3708, 3711, 3712, 3745, 3764, 3786, 3822, 3840, 3844, 3882, 3884, 3893, 3897, 3899, 3902, 3903, 3904, 3913, 3917, 3918, 3944, 3945, 3947, 3950, 3953, 3954, 7290, 7291, 7293, 7295, 7592, 7593, 7608, 9890, 9891, 9892, 9910, 9911, 9920, 9930, 9943, 9945, 9946, 9952, 9957, 9959, 9960, 9964, 9965, 9966, 9971, 9972, 9974, 9976, 9977, 9978, 9981, 9983, 9985, 9986, 10090, 10091, 10093, 10101, 10106, 14952, 14953, 14954, 14963, 14965, 14969, 14970, 34539, 52950, 75127, 75144, 125920, 125922, 145827, 146086, 146089, 146090, 146091, 146093, 146094, 146095, 146096, 146133, 146195, 146199, 146212, 146606, 146688, 146800, 146806, 146809, 146817, 146818, 146819, 146820, 146821, 146822, 146824, 146825, 167039, 167042, 167043, 167044, 167045, 167050, 167083, 167112, 167119, 167120, 167121, 167124, 167125, 167130, 167140, 167141, 168294, 168298, 168329, 168330, 168331, 168332, 168335, 168337, 168338, 168339, 168350, 168739, 168757, 168784, 168868, 168882, 168908, 168909, 168910, 168911, 168912, 189354, 189355, 189773, 189783, 189785, 189877, 189922, 190137, 190146, 190392, 190410, 190411, 190412, 190424, 211687, 211715, 211717, 211719, 211979, 211983, 211986, 317599, 317601, 359953, 359954, 359955, 359956, 359957, 359958, 359959, 359960, 359961, 359962, 359963, 359964, 359965, 359966, 359967, 359968, 359969, 359970, 359971, 359972, 359973, 359974, 359975, 359976, 359977, 359979, 359980, 359981, 359982, 359983, 359984, 359985, 359986, 359987, 359988, 359989, 359990, 359991, 359992, 359993, 359994, 360112, 360113, 360114, 360975]
   environments = Environments.from_openml(task_id=benchmark_tasks, cat_as_str=True, take=4000).scale().shuffle(1)

   #ids          = [457, 163, 339, 1517, 1518, 342, 340, 468, 42793, 1516, 1081, 1079, 1109, 1102, 62, 328, 327, 1465, 1080, 1513, 685, 42046, 42186, 42016, 42071, 42026, 41568, 42056, 42003, 61, 42700, 42098, 42066, 42011, 43859, 43875, 42041, 41583, 41511, 42021, 41997, 1413, 42051, 42036, 41950, 42261, 42031, 1115, 48, 338, 329, 1520, 187, 1106, 285, 1512, 388, 9, 1500, 1499, 952, 41, 1083, 1523, 41939, 7, 42544, 452, 41981, 694, 397, 39, 171, 1482, 42585, 1514, 35, 460, 1088, 41083, 475, 1551, 1508, 387, 1554, 41919, 313, 1515, 41084, 377, 11, 42, 383, 1553, 188, 1549, 1472, 469, 458, 54, 400, 386, 394, 385, 1233, 1555, 307, 40971, 392, 679, 401, 40966, 1468, 1552, 23, 181, 1457, 391, 398, 1501, 1493, 1491, 1492, 395, 40982, 12, 14, 40979, 22, 20, 18, 16, 1466, 36, 40984, 389, 1548, 42532, 473, 393, 396, 1041, 183, 41004, 60, 1525, 1497, 28, 4552, 41003, 41002, 1475, 182, 382, 300, 41164, 41972, 41082, 390, 42140, 375, 41165, 41163, 372, 1459, 1478, 1044, 32, 399, 1568, 26, 1476, 1477, 6, 1481, 184, 41081, 40985, 42141, 41989, 41988, 41986, 41990, 255, 119, 41166, 40927, 41169, 41982, 40996, 554, 42345, 41168, 42396, 180, 41039, 265, 133, 1509, 1483, 1503, 41991, 41167, 41960, 1400, 1596, 150, 42468, 253, 117, 1384, 1394, 247, 1395, 1379, 1378, 159, 1398, 116, 1380, 1381, 1383, 1382, 1385, 1186, 160, 74, 248, 1396, 1397, 147, 1386, 1393, 1209, 156, 1389, 1391, 75, 158, 1401, 1177, 1387, 1399, 1392, 154, 42718, 130, 1390, 1388, 250, 127, 1183, 268, 1214, 261, 78, 134, 118, 254, 123, 272, 148, 157, 271, 129, 263, 115, 252, 141, 1185, 149] 
   #log          = "./study1/outcomes/full-openml-class-12.log.gz"
   #environments = Environments.from_openml(data_id=ids, cat_as_str=True, take=4000).scale()
   #Experiment(environments, [], environment_task=ClassEnvironmentTask()).config(**config).evaluate(log)

   #seed = list(range(4))
   #seed = 1
   #environments = Environments.from_kernel_synthetic(2000, n_context_features=3, n_action_features=0 ,seed=seed).scale(shift='min',scale='minmax',target='rewards').binary()

   result = Experiment(environments, learners, environment_task=SimpleEnvironmentTask(), evaluation_task=SlimOnlineOnPolicyEvalTask()).config(**config).evaluate(log)
   result.filter_fin().plot_learners(y='reward')
