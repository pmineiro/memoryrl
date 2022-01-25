from itertools import tee
from typing import Tuple, Iterable

from learners import MemorizedLearner
from coba.learners import Learner
from coba.environments import Interaction
from coba.experiments import EvaluationTask, OnlineOnPolicyEvalTask

class RewardLoggingEvaluationTask(EvaluationTask):

    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[dict]:

        a,b = tee(interactions)
        d   = OnlineOnPolicyEvalTask().process(learner,b)

        for interaction, eval_result in zip(a,d):
            eval_result["correct_action"] = interaction.kwargs["rewards"].index(1)
            yield eval_result

class FinalPrintEvaluationTask(EvaluationTask):

    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[dict]:

        d = list(OnlineOnPolicyEvalTask().process(learner,interactions))

        # if isinstance(learner, MemorizedLearner):
        #     print(f"s: {learner._cmt.f.t}")
        #     print(f"r: {[n.g.t for n in learner._cmt.nodes if n.g]}")

        return d