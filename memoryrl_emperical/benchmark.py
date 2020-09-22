from memorized import MemorizedLearner
from coba.benchmarks import UniversalBenchmark
from coba.analysis import Plots
from coba.learners import RandomLearner, EpsilonLearner

learner_factories = [
    lambda: MemorizedLearner(.05, 50),
    lambda: MemorizedLearner(.05, 100),
    lambda: MemorizedLearner(.05, 500),
    lambda: MemorizedLearner(.05, 1000),
    lambda: MemorizedLearner(.05, 2000),
    lambda: MemorizedLearner(.05, 50000),
    lambda: MemorizedLearner(.10, 50),
    lambda: MemorizedLearner(.10, 100),
    lambda: MemorizedLearner(.10, 500),
    lambda: MemorizedLearner(.10, 1000),
    lambda: MemorizedLearner(.10, 2000),
    lambda: MemorizedLearner(.10, 50000),
    lambda: RandomLearner(),
    lambda: EpsilonLearner(0.05)
]

result = UniversalBenchmark.from_file("./memoryrl/benchmark.json").evaluate(learner_factories, "./memoryrl/benchmark.log")

Plots.standard_plot(result)