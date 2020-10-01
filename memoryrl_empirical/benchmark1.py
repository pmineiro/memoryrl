from memorized import MemorizedLearner_1
from coba.benchmarks import UniversalBenchmark
from coba.analysis import Plots
from coba.learners import RandomLearner, EpsilonLearner, UcbTunedLearner

learner_factories = [
    lambda: RandomLearner(),
    lambda: EpsilonLearner(0.05),
    lambda: UcbTunedLearner(),
    lambda: MemorizedLearner_1(.05, 100),
    lambda: MemorizedLearner_1(.05, 10000),
    lambda: MemorizedLearner_1(.05, 100000),
]

result = UniversalBenchmark.from_file("./memoryrl_empirical/benchmark_short.json").ignore_raise(False).evaluate(learner_factories)
#result = UniversalBenchmark.from_file("./memoryrl_empirical/benchmark.json").evaluate(learner_factories, "./memoryrl_empirical/benchmark1.log")

Plots.standard_plot(result)