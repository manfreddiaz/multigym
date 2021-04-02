import abc
from abc import ABC


class Arena(abc.ABC):

    @abc.abstractmethod
    def goals(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def spawns(self):
        raise NotImplementedError()


class RandomArena(Arena, ABC):

    def __init__(self):
        pass

    def goals(self):
        pass

    def spawns(self):
        pass


