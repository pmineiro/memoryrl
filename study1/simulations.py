import math
from typing import Tuple, Iterable
from numbers import Number

import numpy as np
from scipy.spatial.distance import cdist

from coba.pipes import Filter
from coba.simulations import LambdaSimulation, Interaction, SimulationFilter
from coba.registry import coba_registry_class
from coba.random import CobaRandom

@coba_registry_class("features_scaled_to_zero_one")
class EuclidNormed(SimulationFilter):
    
    @property
    def params(self):
        return {"zero_one": True}

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        
        materialized_interactions = list(interactions)

        feature_max = {}
        feature_min = {}

        for interaction in materialized_interactions:
            context     = interaction.context
            keys_values = context.items() if isinstance(context,dict) else enumerate(context)
            
            for k,v in keys_values:
                if isinstance(v,Number):
                    feature_max[k] = max(feature_max.get(k,-math.inf),v)
                    feature_min[k] = min(feature_min.get(k, math.inf),v)

        for interaction in materialized_interactions:
            context     = interaction.context
            keys_values = context.items() if isinstance(context,dict) else enumerate(context)
            new_context = {} if isinstance(interaction.context,dict) else [0]*len(context)

            for k,v in keys_values:
                if isinstance(v,Number):
                    if feature_max[k]!=feature_min[k]:
                        new_context[k] = (v-feature_min[k])/(feature_max[k]-feature_min[k])
                else:
                    new_context[k] = v

            if 'reward' in interaction.results and len(interaction.results) == 1:
                yield Interaction(new_context, interaction.actions, interaction.reveals)
            else:
                yield Interaction(new_context, interaction.actions, reveals=interaction.reveals, **interaction.results)

@coba_registry_class("bernoulli_flip")
class BernoulliLabelNoise(Filter[Iterable[Interaction],Iterable[Interaction]]):
    
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

            yield Interaction(interaction.context, interaction.actions, reveals=noised_labels, reward=interaction.reveals)

class MemorizableSimulation(LambdaSimulation):

    def __init__(self, n_interactions:int = 1000, n_features:int = 100, n_anchors=1000, n_actions:int = 10, density:int=1, seed:int = 1) -> None:

        np.random.seed([seed])
        anchors       = np.random.rand(n_anchors,n_features) * density
        anchor_values = np.zeros((n_anchors,n_actions))
        anchor_values[np.arange(n_anchors),np.random.randint(0,n_actions,n_anchors)] = 1

        contexts       = np.random.rand(n_interactions, n_features)*density
        distances      = cdist(contexts,anchors)
        anchor_indexes = np.argmin(distances,axis=1)

        contexts = [tuple(c) for c in contexts]
        actions  = [tuple(l) for l in np.eye(n_actions)]

        def context_generator(index:int):
            return contexts[index]
        
        def action_generator(index:int, context:Tuple[float,...]):
            return actions

        def reward_function(index:int, context:Tuple[float,...], action: Tuple[int,...]):
            return anchor_values[anchor_indexes[index], action.index(1)]

        super().__init__(n_interactions, context_generator, action_generator, reward_function)
    
    def __repr__(self):
        return "memorizable"
