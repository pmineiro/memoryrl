from memorized import MemorizedLearner_1
from coba.benchmarks import UniversalBenchmark, LearnerFactory
from coba.analysis import Plots
from coba.learners import UcbTunedLearner, VowpalLearner
import re

learner_factories = [
    LearnerFactory(MemorizedLearner_1, 0.1, 100),
    LearnerFactory(MemorizedLearner_1, 0.1, 500),
    LearnerFactory(VowpalLearner, epsilon=0.1),
    LearnerFactory(UcbTunedLearner),
]

processes = 20 
maxtasksperchild = 1
json = "./mrucker/benchmark_longish.json"
log = re.sub('json$', 'log', json)

if __name__ == '__main__':
    result = UniversalBenchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).evaluate(learner_factories, log)
    Plots.standard_plot(result)
