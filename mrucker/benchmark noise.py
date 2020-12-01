from memorized import MemorizedLearner_1, ResidualLearner_1
from noisy import BernoulliNoiseLearner
from coba.benchmarks import Benchmark
from coba.learners import UcbTunedLearner, VowpalLearner
import re


base_learners = [
    ResidualLearner_1(0.1, 100),
    MemorizedLearner_1(0.1, 100),
    ResidualLearner_1(0.1, 200),
    MemorizedLearner_1(0.1, 200),
    VowpalLearner(seed=10, epsilon=0.1),
    UcbTunedLearner(),
]

noise_levels = [0.0, 0.1, 0.25]
noisy_learners = []

for noise_level in noise_levels:
    for base_learner in base_learners:
        noisy_learners.append(BernoulliNoiseLearner(base_learner, noise_level))

processes = 3
maxtasksperchild = None
json = "./mrucker/benchmark_medish.json"
log = re.sub('\.json$', '_noise.log', json)

if __name__ == '__main__':
    Benchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).evaluate(noisy_learners, log).standard_plot()
