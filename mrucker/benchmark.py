from memorized import MemorizedLearner_1
from coba.benchmarks import UniversalBenchmark
from coba.analysis import Plots
from coba.learners import RandomLearner, EpsilonLearner, UcbTunedLearner, VowpalLearner
import re

learner_factories = [
    lambda: MemorizedLearner_1(0.1, 100),
    lambda: MemorizedLearner_1(0.1, 500),
    lambda: VowpalLearner(bag=5),
    lambda: UcbTunedLearner(),
#    lambda: EpsilonLearner(0.05),
#    lambda: RandomLearner(),
]

max_processes = 20
json = "./mrucker/benchmark_medish.json"
log = re.sub('json$', 'log', json)

if __name__ == '__main__':
    result = UniversalBenchmark.from_file(json).core_count(max_processes).evaluate(learner_factories, log)
    Plots.standard_plot(result)
