from itertools import tee
from typing import Tuple, Iterable

from matplotlib import pyplot as plt

from learners import MemorizedLearner1
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

        # if isinstance(learner,MemorizedLearner):
        #     learner._cmt.leaf_by_key.items()

        #     for node in learner._cmt.nodes:
        #         if node and node.memories:
        #             x,y = tuple(zip(*[ (k.context[0],k.action[0]) for k in node.memories.keys()]))
        #             plt.scatter(x,y)

        #     plt.show()

        #if isinstance(learner, MemorizedLearner1) and hasattr(learner._cmt, 'root') and learner._cmt.root and learner._cmt.root.left:
        #    print(f"{learner._cmt.d} {learner._cmt.alpha} -- {learner._cmt.root.left.n} -- {learner._cmt.root.right.n}")

        return d

class SlimOnlineOnPolicyEvalTask:
    
    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[dict]:

        for d in OnlineOnPolicyEvalTask().process(learner,interactions):
            d.pop('max_reward')
            d.pop('min_reward')
            d.pop('min_rank')
            d.pop('max_rank')
            d.pop('rank')
            
            yield d