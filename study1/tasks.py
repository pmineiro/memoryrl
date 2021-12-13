from itertools import tee
from typing import Tuple, Iterable

from coba.learners import Learner
from coba.environments import Interaction
from coba.experiments import EvaluationTask, OnPolicyEvaluationTask

class RewardLoggingEvaluationTask(EvaluationTask):

    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[dict]:

        a,b = tee(interactions)
        d   = OnPolicyEvaluationTask().process(learner,b)

        for interaction, eval_result in zip(a,d):
            eval_result["correct_action"] = interaction.kwargs["rewards"].index(1)
            yield eval_result