
from collections import Counter
from itertools import chain, compress
from typing import List, Sequence, Dict

from coba.data.encoders import OneHotEncoder
from coba.data.sources import Source
from coba.simulations import Interaction, Reward, Simulation, MemorySimulation
from coba.random import CobaRandom

class MultiLabelReward(Reward):

    def __init__(self, labels: Dict[int,Sequence[int]]) -> None:
        self._labels = labels

    def observe(self, choices) -> Sequence[float]:
        rewards = []

        for key,action in choices:
            rewards.append(float(action in self._labels[key]))

        return rewards

class MediamillSource(Source[Simulation]):

    def __init__(self, filename:str, balance=True) -> None:

        self._filename = filename
        self._balance = balance

    def read(self) -> Simulation:

        interactions       : List[Interaction] = []
        interactions_labels: Dict[int, Sequence[int]] = {}

        with open(self._filename) as fs:
            
            n_actions = int(next(fs).split(' ')[2])

            action_encoder = OneHotEncoder(list(range(n_actions)))
            actions        = action_encoder.encode(list(range(n_actions)))

            for i,line in enumerate(fs):
                
                items = line.split(' ')
                
                if items[0] == '': continue

                example_labels   = [ action_encoder.encode([int(l)])[0] for l in items[0].split(',') ]
                example_features = { int(item.split(":")[0]):float(item.split(":")[1]) for item in items[1:] }

                interactions.append(Interaction(example_features, actions, i))
                interactions_labels[i] = example_labels

        if self._balance:

            random = CobaRandom(1337)

            label_counts = Counter([l.index(1) for l in chain.from_iterable(interactions_labels.values())])
            top_6 = [k for k, v in sorted(label_counts.items(), key=lambda i: i[1], reverse=True)][0:6]

            for interaction in interactions:
                interactions_labels[interaction.key] = [ l for l in interactions_labels[interaction.key] if l.index(1) not in top_6 ]

            interactions = [i for i in interactions if interactions_labels[i.key]]

            for interaction in interactions:
                interactions_labels[interaction.key] = [random.choice(interactions_labels[interaction.key])]

        return MemorySimulation(interactions, MultiLabelReward(interactions_labels))
    
    def __repr__(self):
        return "Mediamill"