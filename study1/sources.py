
from typing import List, Sequence, Dict

from coba.data.encoders import OneHotEncoder
from coba.data.sources import Source
from coba.simulations import Interaction, Reward, Simulation, MemorySimulation

class MultiLabelReward(Reward):

    def __init__(self, labels: Dict[int,Sequence[int]]) -> None:
        self._labels = labels

    def observe(self, choices) -> Sequence[float]:
        rewards = []

        for key,action in choices:
            rewards.append(float(action in self._labels[key]))

        return rewards

class MediamillSource(Source[Simulation]):

    def __init__(self, filename:str) -> None:

        self._filename = filename

    def read(self) -> Simulation:

        interactions: List[Interaction] = []
        labels      : Dict[int, Sequence[int]] = {}

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
                labels[i] = example_labels

        return MemorySimulation(interactions, MultiLabelReward(labels))
    
    def __repr__(self):
        return "Mediamill"