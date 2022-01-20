from code import interact
from typing import Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD

from coba.encodings import OneHotEncoder
from coba.environments import LambdaSimulation, EnvironmentFilter, SimulatedInteraction
from coba.random import CobaRandom


class LocalSyntheticSimulation(LambdaSimulation):
    """A simple simulation useful for debugging learning algorithms. 

        The simulation's rewards are determined by the location of given context and action pairs with respect to a 
        small set of pre-generated exemplar context,action pairs. Location is currently defined as equality though 
        it could potentially be extended to support any number of metric based similarity kernels. The "local" in 
        the name is due to its close relationship to 'local regression'.
    """

    def __init__(self,
        n_examples_per: int = 500,
        n_contexts: int = 200,
        n_context_feats: int = 2,
        n_actions: int = 10,
        seed: int = 1) -> None:
        """Instantiate a LocalSyntheticSimulation.

        Args:
            n_interactions: The number of interactions the simulation should have.
            n_contexts: The number of unique contexts the simulation should contain.
            n_context_feats: The number of features each interaction context should have.
            n_actions: The number of actions each interaction should have.
            seed: The random number seed used to generate all contexts and action rewards.
        """

        self.args = (n_examples_per, n_contexts, n_context_feats, n_actions, seed)

        self._n_interactions     = n_examples_per
        self._n_context_features = n_context_feats
        self._n_contexts         = n_contexts
        self._n_actions          = n_actions
        self._seed               = seed

        rng = CobaRandom(self._seed)
        np.random.seed(self._seed)

        vec = np.random.randn(n_contexts, n_context_feats)
        vec /= np.linalg.norm(vec, axis=0)

        contexts = [ tuple(v) for v in vec ]
        actions  = OneHotEncoder().fit_encodes(range(n_actions))
        rewards  = {}

        sim_contexts = []
        sim_rewards  = {}

        for c in contexts:
            sim_contexts.extend([c]*n_examples_per)

        for context in contexts:
            for action in actions:
                rewards[(context,action)] = rng.random()
        
        for i,c in enumerate(sim_contexts):
            for action in actions:
                sim_rewards[(i,action)] = rewards[(c,action)]

        def context_generator(index:int, rng: CobaRandom):
            #return sim_contexts[index]
            return tuple([ c + 0.01*rng.random() for c in sim_contexts[index] ])

        def action_generator(index:int, context:Tuple[float,...], rng: CobaRandom):
            return actions

        def reward_function(index:int, context:Tuple[float,...], action: Tuple[int,...], rng: CobaRandom):
            return sim_rewards[(index,action)]

        return super().__init__(n_examples_per*n_contexts, context_generator, action_generator, reward_function, seed)

    @property
    def params(self):
        return { 
            "n_A"    : self._n_actions,
            "n_C"    : self._n_contexts,
            "n_C_phi": self._n_context_features,
            "seed"   : self._seed
        }

    def __str__(self) -> str:
        return f"LocalSynth(A={self._n_actions},C={self._n_contexts},c={self._n_context_features},seed={self._seed})"

    def __reduce__(self) -> Tuple[object, ...]:
        return (LocalSyntheticSimulation, self.args)

class MNIST_LabelFilter(EnvironmentFilter):

    def __init__(self, labels) -> None:
        self._labels = labels

    @property
    def params(self):
        return {'labels': self._labels}

    def filter(self, interactions):
        for interaction in interactions:
            if interaction.actions[interaction.kwargs["rewards"].index(1)] in self._labels:
                
                label_indexes = [ interaction.actions.index(label) for label in self._labels ]
                rewards       = [ interaction.kwargs["rewards"][i] for i in label_indexes ]

                yield SimulatedInteraction(interaction.context, self._labels, rewards=rewards)

class MNIST_SVD(EnvironmentFilter):

    def __init__(self, rank, seed=42) -> None:
        self._rank = rank
        self._seed = seed

    @property
    def params(self):
        return {'rank': self._rank, 'svd_seed': self._seed}

    def filter(self, interactions):        
        rows = []
        stuff = []

        for interaction in interactions:
            rows.append(interaction.context)
            stuff.append((interaction.actions, interaction.kwargs["rewards"]))

        svd = TruncatedSVD(n_components=self._rank, n_iter=5, random_state=self._seed)
        Xprimes = svd.fit_transform(np.array(rows))

        for xprime, (actions, rewards) in zip(Xprimes, stuff):
            yield SimulatedInteraction( tuple(xprime), actions, rewards=rewards)