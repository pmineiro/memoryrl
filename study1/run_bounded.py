from memory import EMT
from learners import StackedLearner
from tasks import SlimOnlineOnPolicyEvalTask

from coba.learners import VowpalEpsilonLearner
from coba.environments import Environments
from coba.experiments  import Experiment, SimpleEnvironmentTask

config  = {"processes": 8, "chunk_by":'task', 'maxchunksperchild': 2 }
epsilon = 0.1

if __name__ == '__main__':
    learners = [
        VowpalEpsilonLearner(epsilon, features=["a","xa","xxa"]),
        StackedLearner      (epsilon, EMT(bound=1000 , scorer=3, router=2, split=100,  interactions=['xa']), "xxa", False, True, False),
        StackedLearner      (epsilon, EMT(bound=2000 , scorer=3, router=2, split=100,  interactions=['xa']), "xxa", False, True, False),
        StackedLearner      (epsilon, EMT(bound=16000, scorer=3, router=2, split=100,  interactions=['xa']), "xxa", False, True, False),
        StackedLearner      (epsilon, EMT(bound=32000, scorer=3, router=2, split=100,  interactions=['xa']), "xxa", False, True, False),
    ]

    environments = Environments.from_template("./study1/experiments/fixed.json", n_shuffle=50, n_take=32000)
    description = "First pass at bounded experiments dataset."
    log         = "./study1/outcomes/ICLR-2023-bounded.log.gz"

    environments = sorted(environments, key=lambda e: (e.params['shuffle'],e.params['openml_task']))

    result = Experiment(environments, learners, description, environment_task=SimpleEnvironmentTask(), evaluation_task=SlimOnlineOnPolicyEvalTask()).config(**config).evaluate(log)
    result.filter_fin().plot_learners(y='reward')
