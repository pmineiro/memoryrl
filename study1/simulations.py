from itertools import count
from typing import Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD

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
        seed: int = 2) -> None:
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

        if n_context_feats == 1:
            contexts = list(map(float,np.random.randn(n_contexts)))
        else:
            vec = np.random.randn(n_contexts, n_context_feats)
            vec /= np.linalg.norm(vec, axis=0)
            contexts = [ tuple(map(float,v)) for v in vec ]
 
        action_sets = [ list(map(float,np.random.randn(2))) for _ in range(n_contexts) ]
        rewards     = {}

        sim_contexts    = []
        sim_action_sets = []
        sim_rewards     = []

        noise = iter((.08*rng.random() for _ in count()))

        for context, action_set in zip(contexts,action_sets):

            rewards = rng.randoms(len(action_sets))

            for _ in range(n_examples_per):
                
                noisy_context = context+next(noise) if n_context_feats==1 else tuple([c+next(noise) for c in context])
                noisy_action_set = [ a +next(noise) for a in action_set ]

                sim_contexts.append(noisy_context)
                sim_action_sets.append(noisy_action_set)
                sim_rewards.append(dict(zip(noisy_action_set,rewards)))

        def context_generator(index):
            return sim_contexts[index]

        def action_generator(index, context):
            return sim_action_sets[index]

        def reward_function(index, context, action):
            return sim_rewards[index][action]

        return super().__init__(n_examples_per*n_contexts, context_generator, action_generator, reward_function)

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