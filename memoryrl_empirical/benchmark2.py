from memorized import MemorizedLearner_1, MemorizedLearner_1
from coba.benchmarks import UniversalBenchmark
from coba.analysis import Plots
from coba.learners import RandomLearner, EpsilonLearner, UcbTunedLearner

learner_factories = [
    lambda: RandomLearner(),
    lambda: EpsilonLearner(0.05),
    lambda: UcbTunedLearner(),
    lambda: MemorizedLearner_1(.05, 10000),
    lambda: MemorizedLearner_2(.05, 10000),
]

result = UniversalBenchmark.from_file("./memoryrl_empirical/benchmark.json").evaluate(learner_factories, "./memoryrl_empirical/benchmark.log")

Plots.standard_plot(result)