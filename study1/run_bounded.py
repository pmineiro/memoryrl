from memory import EMT_VW
from learners import ComboLearner
from tasks import SlimOnlineOnPolicyEvalTask

from coba.environments import Environments
from coba.experiments  import Experiment, SimpleEnvironmentTask

config  = {"processes": 8, "chunk_by":'source', 'maxchunksperchild': 0 }
epsilon = 0.1

if __name__ == '__main__':
    learners = [
        ComboLearner (epsilon, EMT_VW(eigen=True , bound=2000 , scorer=3, router=2, split=100,  interactions=['xa']), "xxa", False, True),
        ComboLearner (epsilon, EMT_VW(eigen=True , bound=4000 , scorer=3, router=2, split=100,  interactions=['xa']), "xxa", False, True),
        ComboLearner (epsilon, EMT_VW(eigen=True , bound=8000 , scorer=3, router=2, split=100,  interactions=['xa']), "xxa", False, True),
        ComboLearner (epsilon, EMT_VW(eigen=True , bound=16000, scorer=3, router=2, split=100,  interactions=['xa']), "xxa", False, True),
        ComboLearner (epsilon, EMT_VW(eigen=True , bound=24000, scorer=3, router=2, split=100,  interactions=['xa']), "xxa", False, True),
        ComboLearner (epsilon, EMT_VW(eigen=True , bound=-1   , scorer=3, router=2, split=100,  interactions=['xa']), "xxa", False, True),
    ]

    environments = Environments.from_template("./study1/experiments/fixed.json", n_shuffle=1, n_take=32000)
    description = "First pass at bounded experiments dataset."
    log         = "./study1/outcomes/neurips-bounded.log.gz"

    result = Experiment(environments, learners, description, environment_task=SimpleEnvironmentTask(), evaluation_task=SlimOnlineOnPolicyEvalTask()).config(**config).evaluate(log)
    result.filter_fin().plot_learners(y='reward')
