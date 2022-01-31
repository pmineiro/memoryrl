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
log        = None#"./study1/outcomes/mnist_4.log.gz"
config     = {"processes": 8, "chunk_by":'task' }

epsilon      = 0.1

if __name__ == '__main__':

   XLs      = [ [] ]
   XSs      = [ [] ]
   cs       = [ ConstSplitter(100) ]
   ds       = [ 2 ]
   alphas   = [ .25 ]
   init_ws  = [ 0 ]
   sgds     = [ 'coin' ]
   bases    = [ 'exp' ]
   lrs      = [ .01 ]
   l2s      = [ .5 ]

   # l2=.5, lr=.01, bases='l1', sgd='coin', c=1000, d=1, alpha=.25, init_w=1 (best so far)

   #I still don't have a good understanding of the various update strategies for splitting

   #update router update strategies
      #pure supervised feedback
      #prediction error feedback
      #prediction score feedback

   #insert router update strategies
      #prediction error feedback
      #prediction score feedback

   #splitting router update strategies
      #prediction error feedback
      #prediction score feedback

   #insert scorer update strageties
      #do nothing...
      #

   #updating on insert using mem_val differences gives us a faster start out of the gate
   #adding in an l2 term prevents us from cratering...
   #low learning rate also seems to help...

   learners = [
      VowpalEpsilonLearner(epsilon),
      MemorizedLearner(epsilon, CMT(6000, None, MetricScorer("l1"), c=ConstSplitter(20000), d=1, alpha=None)),
   ]

   learners.append(MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,[],True,0), RankScorer(0,[],0,"exp",.01,.5,"coin"), c=cs[0], d=2, alpha=0.25, v=(2,1))))
   learners.append(MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,[],True,0), RankScorer(0,[],0,"exp",.01,.5,"coin"), c=cs[0], d=2, alpha=0.25, v=(1,1))))
   learners.append(MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,[],True,0), RankScorer(0,[],0,"exp",.01,.5,"coin"), c=cs[0], d=2, alpha=0.25, v=(2,2))))
   learners.append(MemorizedLearner(epsilon, CMT(6000, LogisticRouter(0,[],True,0), RankScorer(0,[],0,"exp",.01,.5,"coin"), c=cs[0], d=2, alpha=0.25, v=(1,2))))

   #environments = Environments([LocalSyntheticSimulation(20, n_context_feats=1, n_actions=2, n_contexts=50)]).binary().shuffle([2]).take(1000)
   environments = Environments.from_openml(554, cat_as_str=True).filter(MNIST_LabelFilter(['9','4'])).filter(MNIST_SVD(30)).scale("min","minmax").shuffle([1])

   Experiment(environments, learners, environment_task=SimpleEnvironmentTask(), evaluation_task=FinalPrintEvaluationTask()).config(**config).evaluate(log).filter_fin().plot_learners(sort="reward")
