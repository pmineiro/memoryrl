from memorized import MemorizedLearner_1
from coba.benchmarks import UniversalBenchmark
from coba.analysis import Plots
from coba.learners import RandomLearner, EpsilonLearner, UcbTunedLearner, VowpalLearner

learner_factories = [
    lambda: MemorizedLearner_1(0.1, 100),
    lambda: MemorizedLearner_1(0.1, 500),
    lambda: VowpalLearner(bag=5),
    lambda: UcbTunedLearner(),
#    lambda: EpsilonLearner(0.05),
#    lambda: RandomLearner(),
]

#result = UniversalBenchmark.from_file("./mrucker/benchmark_short.json").ignore_raise(False).evaluate(learner_factories, "./mrucker/benchmark_short.log")
#result = UniversalBenchmark.from_file("./mrucker/benchmark_long.json").evaluate(learner_factories, "./mrucker/benchmark_long.log")
result = UniversalBenchmark.from_file("./mrucker/benchmark_medish.json").evaluate(learner_factories, "./mrucker/benchmark_medish.log")
#result = UniversalBenchmark.from_file("./mrucker/benchmark_smallish.json").evaluate(learner_factories, "./mrucker/benchmark_smallish.log")

Plots.standard_plot(result)
