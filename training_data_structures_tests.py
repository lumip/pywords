# pywords - A machine learning implementation for words transformations in natural languages (e.g. verb conjugations) using decision trees
# Copyright (C) 2017  Lukas Prediger <lukas.prediger@rwth-aachen.>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>

import unittest

from training_data_structures import TrainingSetElement, Cluster, FrozenCluster, ClusterSet

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
