import os
import itertools

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'    ] = '1'
os.environ['OMP_NUM_THREADS'     ] = '1'

from memory import CMT
from learners import MemorizedLearner
from routers import LogisticRouter, RandomRouter
from scorers import RankScorer, MetricScorer
from tasks import FinalPrintEvaluationTask
from simulations import LocalSyntheticSimulation, MNIST_LabelFilter, MNIST_SVD
from splitters import LogSplitter, ConstSplitter

from coba.environments import Environments
from coba.experiments  import Experiment, ClassEnvironmentTask, SimpleEnvironmentTask, OnlineOnPolicyEvalTask
from coba.learners     import VowpalEpsilonLearner, CorralLearner, EpsilonBanditLearner

experiment = 'full6'
json       = f"./study1/experiments/{experiment}.json"
log        = None#"mnist_2.log.gz"#f"./study1/outcomes/full6_22.log.gz"
config     = {"processes": 8, "chunk_by":'task' }

epsilon      = 0.1

if __name__ == '__main__':

   XLs      = [ [] ]
   XSs      = [ [] ]
   cs       = [ ConstSplitter(1000) ]
   ds       = [ 1 ]
   alphas   = [ .25 ]
   init_ws  = [ 0 ]
   sgds     = [ 'coin' ]
   bases    = [ "exp" ]
   lrs      = [ .01 ]
   l2s      = [ .75 ]
   bls      = [ 'internal' ]

   # l2=.5, lr=.01, bases='l1', sgd='coin', c=1000, d=1, alpha=.25, init_w=1 (best so far)

   #updating on insert using mem_val differences gives us a faster start out of the gate
   #adding in an l2 term prevents us from cratering...
   #low learning rate also seems to help...

   # learners = [
   #    EpsilonBanditLearner(epsilon)
   # ]

   learners = [
      VowpalEpsilonLearner(epsilon),
      MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,[],True,0.25), MetricScorer("l1"), c=ConstSplitter(3000), d=1, alpha=0.70)),
   ]

   # for c in cs:
   #    learners.append(MemorizedLearner(epsilon, CMT(6000, RandomRouter(), MetricScorer("l2"), c=c, d=1, alpha=0.25)))

   for c, d, alpha, XL, XS, w, base, lr, l2, sgd, bl in itertools.product(cs,ds, alphas, XLs, XSs, init_ws, bases, lrs, l2s, sgds, bls):
      learners.append(MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,[],True,0), RankScorer(0,XS,w,base,lr,l2,sgd,bl), c=c, d=d, alpha=alpha, v=1)))
      learners.append(MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,[],True,0), RankScorer(0,XS,w,base,lr,l2,sgd,bl), c=c, d=d, alpha=alpha, v=2)))
      learners.append(MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,[],True,0), RankScorer(0,XS,w,base,lr,l2,sgd,bl), c=c, d=d, alpha=alpha, v=3)))

   #environments = Environments([LocalSyntheticSimulation(20, n_context_feats=2, n_actions=2, n_contexts=50)]).binary().shuffle(range(1))
   environments = Environments.from_openml(554, cat_as_str=True).filter(MNIST_LabelFilter(['9','4'])).filter(MNIST_SVD(30)).scale("min","minmax").take(1000)

   Experiment(environments, learners, environment_task=ClassEnvironmentTask(), evaluation_task=FinalPrintEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners()
