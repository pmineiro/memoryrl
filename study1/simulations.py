from gzip import GzipFile

from typing import Tuple

from coba.random import CobaRandom
from coba.pipes import Source, MemorySource
from coba.simulations import Simulation, LibsvmSimulation, ManikSimulation, LambdaSimulation
from coba.registry import coba_registry_class

@coba_registry_class("Mediamill")
class MediamillSimulation(Source[Simulation]):

    #The mediamill data set found on the following page
    #http://manikvarma.org/downloads/XC/XMLRepository.html

    def read(self) -> Simulation:
        with GzipFile("./study1/datasets/Mediamill_data.gz", 'r') as fs:
            return ManikSimulation(MemorySource([line.decode('utf-8') for line in fs ])).read()

    def __repr__(self):
        return "mediamill"

@coba_registry_class("Memorizable")
class MemorizableSimulation(Source[Simulation]):
    
    def read(self) -> Simulation:
        
        contexts = [(0,), (1,), (2,)]
        actions  = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        def context_generator(index:int):
            return contexts[index%3]
        
        def action_generator(index:int, context:Tuple[float,...]):
            return actions

        def reward_function(index:int, context:Tuple[float,...], action: Tuple[int,...]):
            return float( actions[context[0]] == action)

        return LambdaSimulation(10000, context_generator, action_generator, reward_function).read()
    
    def __repr__(self):
        return "memorizable"

@coba_registry_class("Sector")
class SectorSimulation(Source[Simulation]):

    #combination of the train and test sets of the sector dataset
    #https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#sector

    def read(self) -> Simulation:
        with GzipFile("./study1/datasets/sector.gz", 'r') as fs:
            return LibsvmSimulation(MemorySource([line.decode('utf-8') for line in fs ])).read()
    
    def __repr__(self):
        return "sector"

@coba_registry_class("Rcv1")
class Rcv1Simulation(Source[Simulation]):

    #the train set of the rcv1.multiclass dataset (train in order to keep the dataset small)
    #https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#rcv1.multiclass

    def read(self) -> Simulation:
        with GzipFile("./study1/datasets/rcv1.gz", 'r') as fs:
            return LibsvmSimulation(MemorySource([line.decode('utf-8') for line in fs ])).read()

    
    def __repr__(self):
        return "rcv1"
