from memorized import MemorizedLearner_1, ResidualLearner_1
from coba.benchmarks import Benchmark
from coba.learners import UcbTunedLearner, VowpalLearner
import re

learners = [
    ResidualLearner_1(0.1, 100),
    MemorizedLearner_1( 0.1, 100),
    ResidualLearner_1(0.1, 200),
    MemorizedLearner_1(0.1, 200),
    VowpalLearner(seed=10, epsilon=0.1),
    UcbTunedLearner(),
]

processes = 3
maxtasksperchild = None
json = "./mrucker/benchmark_short.json"
log = re.sub('json$', 'log', json)

if __name__ == '__main__':
    Benchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).evaluate(learners, log).standard_plot()
