from memorized import MemorizedLearner_1, ResidualLearner_1
from coba.benchmarks import UniversalBenchmark, LearnerFactory
from coba.analysis import Plots
from coba.learners import UcbTunedLearner, VowpalLearner
import re

learner_factories = [
    LearnerFactory(ResidualLearner_1, 0.1, 100),
    LearnerFactory(MemorizedLearner_1, 0.1, 100),
    LearnerFactory(ResidualLearner_1, 0.1, 200),
    LearnerFactory(MemorizedLearner_1, 0.1, 200),
    LearnerFactory(VowpalLearner, seed=10, epsilon=0.1),
#    LearnerFactory(VowpalLearner, seed=10, epsilon=0.1, flags='--first_only'),
#    LearnerFactory(VowpalLearner, seed=10, epsilon=0.1, is_adf=False),
    LearnerFactory(UcbTunedLearner),
]

processes = 20 
maxtasksperchild = 1
json = "./mrucker/benchmark_longish.json"

log = re.sub('json$', 'log', json)

if __name__ == '__main__':
    result = UniversalBenchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).evaluate(learner_factories, log)
    Plots.standard_plot(result)
