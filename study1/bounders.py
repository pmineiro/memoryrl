from abc import ABC, abstractmethod
from typing import Hashable, Sequence

class Bounder(ABC):

    @abstractmethod
    def touch(self, key: Hashable) -> Sequence[Hashable]:
        ...

class NoBounds(Bounder):

    def touch(self, key: Hashable) -> Sequence[Hashable]:
        return []

    def __str__(self) -> str:
        return "None"

class LruBounds(Bounder):

    PREV = 0
    NEXT = 1
    ITEM = 2

    def __init__(self, size:int) -> None:
        self._size = size
        self._dict = {}
        self._root = None

    def touch(self, item: Hashable)-> Sequence[Hashable]:

        if item in self._dict:
            self._put_root(self._pop_node(self._dict[item]))

        if item not in self._dict:
            new_node = self._new_node(item)
            self._dict[item] = new_node
            self._put_root(new_node)

        if len(self._dict) <= self._size: 
            return []
        else:
            lru_node = self._root[LruBounds.PREV]
            lru_item = lru_node[LruBounds.ITEM]

            self._pop_node(lru_node)
            self._dict.pop(lru_item)

            return [lru_item]

    def _new_node(self, item):
        new_node = []
        new_node[:] = [new_node,new_node,item]
        return new_node

    def _pop_node(self, node):

        if self._root is node and self._root[1] is self._root:
            self._root = None 
            
        if self._root is node and self._root[1] is not self._root:
            self._root = self._root[1]

        node[LruBounds.PREV][LruBounds.NEXT] = node[LruBounds.NEXT]
        node[LruBounds.NEXT][LruBounds.PREV] = node[LruBounds.PREV]

        return node

    def _put_root(self, node):
        if self._root is not None:
            root = self._root
            node[LruBounds.PREV]                 = root[LruBounds.PREV]
            node[LruBounds.NEXT]                 = root
            root[LruBounds.PREV][LruBounds.NEXT] = node
            root[LruBounds.PREV]                 = node

        self._root = node

    def __str__(self) -> str:
        return f"LRU({self._size})"
