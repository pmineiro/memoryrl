from memory import EMT_VW
from learners import EpisodicLearner, ComboLearner
from tasks import SlimOnlineOnPolicyEvalTask

from coba.environments import Environments
from coba.learners     import VowpalEpsilonLearner
from coba.experiments  import Experiment, ClassEnvironmentTask

config  = {"processes": 8, "chunk_by":'source', 'maxchunksperchild': 0 }
epsilon = 0.1

if __name__ == '__main__':

   learners = [
      # #Parametric
      VowpalEpsilonLearner(epsilon, features=["a","xa","xxa"]),

      # #EMT-CB (self-consistent)
      EpisodicLearner     (epsilon, EMT_VW(eigen=True , bound=-1, scorer=3, router=2, split=100,  interactions=['xa'])),
      
      # #EMT-CB (not self-consistent)
      EpisodicLearner     (epsilon, EMT_VW(eigen=True , bound=-1, scorer=4, router=2, split=100,  interactions=['xa'])),
      
      # #CMT-CB
      EpisodicLearner     (epsilon, EMT_VW(eigen=False,                                           interactions=['xa'])), 

      # #PEMT
      ComboLearner        (epsilon, EMT_VW(eigen=True , bound=-1, scorer=3, router=2, split=100,  interactions=['xa']), "xxa", False, True),

      # #PCMT
      ComboLearner        (epsilon, EMT_VW(eigen=False, bound=-1,                                 interactions=['xa']), "xxa", False, True),

      # #EMT-CB (self-consistent) (python implementation) (we're not using this in the paper its just here for comparison)
      #EpisodicLearner     (epsilon, EMT(NoBounds(), EigenRouter(method="Oja",features=['x','a']), RankScorer("exp",['x','a','xa']), splitter=ConstSplitter(100), d=0.00, alpha=0)),
      #EpisodicLearner     (epsilon, EMT(NoBounds(), EigenRouter(method="PCA",features=['x','a']), RankScorer("exp",['x','a','xa']), splitter=ConstSplitter(100), d=0.00, alpha=0)),
   ]

   description = "First pass at the final experiment dataset with all necessary learners. All unbounded trees."
   #log         = None#"./study1/outcomes/neurips-2-cmt.log.gz"
   log         = "./study1/outcomes/neurips-2-vw-5.log.gz"

   #environments = Environments.from_template("./study1/experiments/sanity.json", n_shuffle=10, n_take=1000)
   environments = Environments.from_template("./study1/experiments/neurips.json", n_shuffle=1)
   #environments = Environments.from_template("./study1/experiments/fixed.json", n_shuffle=1, n_take=4000)

   result = Experiment(environments, learners, description, environment_task=ClassEnvironmentTask(), evaluation_task=SlimOnlineOnPolicyEvalTask()).config(**config).evaluate(log)
   result.filter_fin().plot_learners(y='reward')
