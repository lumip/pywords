import unittest

from training_data_structures import TrainingSetElement, Cluster, FrozenCluster, ClusterSet

# todo: complete tests

class ClusterSetTests(unittest.TestCase):

    class TestElem:
        def __init__(self, value, hash) -> None:
            self._value = value
            self._hash = hash

        def __hash__(self) -> int:
            return self._hash

        def __eq__(self, other) -> bool:
            return other._value == self._value

        def __repr__(self) -> str:
            return "(" + str(self._value) + " # " + str(self._hash) + ")@" + str(id(self))

    def test_cluster_set(self) -> None:
        c = ClusterSet()
        c.add(self.TestElem(value=1, hash=1))
        c.add(self.TestElem(value=2, hash=2))
        c.add(self.TestElem(value=3, hash=2))
        c.add(self.TestElem(value=2, hash=2))
        expected = frozenset({frozenset({self.TestElem(value=1, hash=1)}),
                              frozenset({self.TestElem(value=2, hash=2), self.TestElem(value=2, hash=2)}),
                              frozenset({self.TestElem(value=3, hash=2)})})
        self.assertEqual(c.get_clusters(), expected)
