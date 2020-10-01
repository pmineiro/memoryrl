from memorized import MemorizedLearner_1
from coba.benchmarks import UniversalBenchmark
from coba.analysis import Plots
from coba.learners import RandomLearner, EpsilonLearner, UcbTunedLearner, VowpalLearner

learner_factories = [
    lambda: RandomLearner(),
    lambda: EpsilonLearner(0.05),
    lambda: UcbTunedLearner(),
    lambda: VowpalLearner(bag=5),
    lambda: MemorizedLearner_1(.05, 100),
    lambda: MemorizedLearner_1(.05, 500),
    lambda: MemorizedLearner_1(.05, 2500),
]

result = UniversalBenchmark.from_file("./mrucker/benchmark_short.json").ignore_raise(False).evaluate(learner_factories)
#result = UniversalBenchmark.from_file("./mrucker/benchmark_long.json").evaluate(learner_factories, "./mrucker/benchmark_long.log")

Plots.standard_plot(result)