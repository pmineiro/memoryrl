from memory import EMT, CMT
from learners import EpisodicLearner, StackedLearner
from tasks import SlimOnlineOnPolicyEvalTask

from coba.environments import Environments
from coba.learners     import VowpalEpsilonLearner
from coba.experiments  import Experiment, ClassEnvironmentTask

config  = {"processes": 8, "chunk_by":'task', 'maxtasksperchunk': None, 'maxchunksperchild': 1 }
epsilon = 0.1

if __name__ == '__main__':

   learners = [
      # #Parametric
      VowpalEpsilonLearner(epsilon, features=["a","xa","xxa"]),

      # #EMT-CB (self-consistent)
      EpisodicLearner     (epsilon, EMT(split=100, scorer=3, router=2, bound=-1,                       interactions=['xa'])),
      
      # #EMT-CB (not self-consistent)
      EpisodicLearner     (epsilon, EMT(split=100, scorer=4, router=2, bound=-1,                       interactions=['xa'])),
      
      # #CMT-CB
      EpisodicLearner     (epsilon, CMT(n_nodes=2000, leaf_multiplier=9 , dream_repeats=10, alpha=0.50, interactions=['xa'])),

      # #PEMT
      StackedLearner      (epsilon, EMT(split=100, scorer=3, router=2, bound=-1,                       interactions=['xa']), "xxa", False, True, False),

      # #PCMT
      StackedLearner      (epsilon, CMT(n_nodes=2000, leaf_multiplier=9, dream_repeats=10, alpha=0.50, interactions=['xa']), "xxa", False, True, False),

      #######TESTING LEARNERS###########
      # #EMT-CB (self-consistent) (python implementation) (we're not using this in the paper its just here for comparison)
      #EpisodicLearner     (epsilon, EMT_PY(NoBounds(), EigenRouter(method="Oja",features=['x','a']), RankScorer("exp",['x','a','xa']), splitter=ConstSplitter(100), d=0.00, alpha=0)),
      #EpisodicLearner     (epsilon, EMT_PY(NoBounds(), EigenRouter(method="PCA",features=['x','a']), RankScorer("exp",['x','a','xa']), splitter=ConstSplitter(100), d=0.00, alpha=0)),
   ]

   description = "Full on 50 replicate run for the ICRL 2023 paper."
   #log         = None#"./study1/outcomes/neurips-2-cmt.log.gz"
   log         = "./study1/outcomes/ICLR-2023-unbounded.log"

   #environments = Environments.from_template("./study1/experiments/sanity.json", n_shuffle=1, n_take=4000)
   environments = Environments.from_template("./study1/experiments/neurips.json")
   #environments = Environments.from_template("./study1/experiments/fixed.json", n_shuffle=1, n_take=4000)

   result = Experiment(environments, learners, description, environment_task=ClassEnvironmentTask(), evaluation_task=SlimOnlineOnPolicyEvalTask()).config(**config).evaluate(log)
   result.filter_fin().plot_learners()
