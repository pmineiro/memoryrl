import random

from gzip import GzipFile
from typing import List, Sequence, Dict, Tuple

from coba.data.encoders import OneHotEncoder
from coba.data.sources import Source
from coba.simulations import Interaction, Reward, Simulation, MemorySimulation, ClassificationSimulation
from coba.simulations.core import LambdaSimulation
from coba.tools.registry import coba_registry_class

import torch

class MultiLabelReward(Reward):

    def __init__(self, labels: Dict[int,Sequence[int]]) -> None:
        self._labels = labels

    def observe(self, choices) -> Sequence[float]:
        rewards = []

        for key,action in choices:
            rewards.append(float(action in self._labels[key]))

        return rewards

@coba_registry_class("Mediamill")
class MediamillSource(Source[Simulation]):

    #The mediamill data set found on the following page
    #http://manikvarma.org/downloads/XC/XMLRepository.html

    def read(self) -> Simulation:

        interactions       : List[Interaction] = []
        interactions_labels: Dict[int, Sequence[int]] = {}

        with GzipFile("./study1/datasets/Mediamill_data.gz", 'r') as fs:

            n_actions = int(next(fs).decode('utf-8').split(' ')[2])

            action_encoder = OneHotEncoder(list(range(n_actions)))
            actions        = action_encoder.encode(list(range(n_actions)))

            for i,line in enumerate(fs):

                items = line.decode('utf-8').split(' ')

                if items[0] == '': continue

                example_labels   = [ action_encoder.encode([int(l)])[0] for l in items[0].split(',') ]
                example_features = { int(item.split(":")[0]):float(item.split(":")[1]) for item in items[1:] }

                interactions.append(Interaction(example_features, actions, i))
                interactions_labels[i] = example_labels

        return MemorySimulation(interactions, MultiLabelReward(interactions_labels))
    
    def __repr__(self):
        return "mediamill"

@coba_registry_class("Memorizable")
class MemorizableSource(Source[Simulation]):
    
    def read(self) -> Simulation:
        
        contexts  = list(map(tuple,torch.randn(100,10).tolist()))
        actions = [ (1, 0, 0), (0, 1, 0), (0, 0, 1) ]
        answers = { context: random.choice(actions) for context in contexts }

        def context_generator(index:int):
            return random.choice(contexts)
        
        def action_generator(index:int, context:Tuple[float,...]):
            return actions

        def reward_function(index:int, context:Tuple[float,...], action: Tuple[int,...]):
            return float(answers[context] == action)

        return LambdaSimulation(10000, context_generator, action_generator, reward_function).read()
    
    def __repr__(self):
        return "memorizable"

class LibSvmSource(Source[Simulation]):
    
    def __init__(self, filename) -> None:
        self._filename = filename

    def read(self) -> Simulation:

        with GzipFile(self._filename, 'r') as fs:

            features = []
            labels   = []

            for line in fs:

                items = line.decode('utf-8').strip().split(' ')

                label    = int(items[0])
                splits   = [ i.split(":") for i in items[1:] ]
                encoded  = [ (int(s[0]), float(s[1])) for s in splits ]
                sparse   = tuple(zip(*encoded))

                features.append(sparse)
                labels.append(label)

        return ClassificationSimulation(features, labels)

@coba_registry_class("Sector")
class SectorSource(LibSvmSource):

    #combination of the train and test sets of the sector dataset
    #https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#sector

    def __init__(self) -> None:
        super().__init__("./study1/datasets/sector.gz")
    
    def __repr__(self):
        return "sector"

@coba_registry_class("Rcv1")
class Rcv1Source(LibSvmSource):

    #the train set of the rcv1.multiclass dataset (train in order to keep the dataset small)
    #https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#rcv1.multiclass

    def __init__(self) -> None:
        super().__init__("./study1/datasets/rcv1.gz")
    
    def __repr__(self):
        return "rcv1"
