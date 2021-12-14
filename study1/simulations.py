import math

from collections import defaultdict
from numbers import Number
from typing import Tuple, Iterable

import numpy as np

from coba.environments import LambdaSimulation, Interaction, EnvironmentFilter, SimulatedEnvironment
from coba.registry import coba_registry_class
from coba.random import CobaRandom

@coba_registry_class("bernoulli_flip")
class BernoulliLabelNoise(EnvironmentFilter):

    def __init__(self, prob=0) -> None:
        self._prob = prob
        self._rng = CobaRandom(1)

    @property
    def params(self):
        return {"flip": self._prob}

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        for interaction in interactions:
            if self._prob > 0 and self._rng.random() <= self._prob:
                #we flip them all, otherwise the chance of us receiving
                #receiving an error for a wrong action selection will be
                #much lower than a right action selection
                noised_labels = [ (1-r) for r in interaction.reveals]
            else:
                noised_labels = [r for r in interaction.reveals]

            yield Interaction(interaction.context, interaction.actions, reveals=noised_labels, reward=interaction.reveals, **interaction.results)

class MemorizableSimulation(SimulatedEnvironment):

    def __init__(self,
        n_interactions:int = 1000,
        n_features:int = 2,
        n_context=3,
        n_actions:int = 10,
        seed:int = 1) -> None:

        self._n_interactions = n_interactions
        self._n_features     = n_features
        self._n_contexts      = n_context
        self._n_actions      = n_actions
        self._seed           = seed

    @property
    def params(self):
        return {"n_int": self._n_interactions, "n_feat": self._n_features, "n_ctx": self._n_contexts, "n_act": self._n_actions }

    def read(self) -> Iterable[Interaction]:

        rng = CobaRandom(self._seed)

        #contexts = [ HashableDict(enumerate(rng.randoms(self._n_features))) for _ in range(self._n_contexts) ]
        contexts = [ tuple(rng.randoms(self._n_features)) for _ in range(self._n_contexts) ]
        
        actions  = [ tuple(l) for l in np.eye(self._n_actions).astype(int)]
        rewards  = defaultdict(int)

        for context in contexts: rewards[(context,rng.choice(actions))] = 1

        def context_generator(index:int):
            return rng.choice(contexts)

        def action_generator(index:int, context:Tuple[float,...]):
            return actions

        def reward_function(index:int, context:Tuple[float,...], action: Tuple[int,...]):
            return rewards[(context,action)]

        return LambdaSimulation(self._n_interactions, context_generator, action_generator, reward_function).read()

    def __repr__(self):
        return f"Memorizable{self.params}".replace("{","(").replace("}",")")
