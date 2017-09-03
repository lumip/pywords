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

import input_parsing

class HangeulComposerTests(unittest.TestCase):

    def test_decompose(self) -> None:
        input = "생각해요"
        composer = input_parsing.HangeulComposer()
        output = composer.decompose(input)
        self.assertEquals("생각해요", output)

    def test_process_input(self) -> None:
        input = "생각해요"
        composer = input_parsing.HangeulComposer()
        output = composer.process_input(input)
        self.assertEquals("생각해요", output)

    def test_compose(self) -> None:
        input = "생각해요"
        composer = input_parsing.HangeulComposer()
        output = composer.compose(input)
        self.assertEquals("생각해요", output)

    def test_process_output(self) -> None:
        input = "생각해요"
        composer = input_parsing.HangeulComposer()
        output = composer.process_output(input)
        self.assertEquals("생각해요", output)


    # todo: tests with uncomposable elements (single elements and non-hangeul letters)

class CombinedProcessorTests(unittest.TestCase):

    class SubstitutionProcessor(input_parsing.WordProcessor):

        def __init__(self, x: str, y: str) -> None:
            self.x = x
            self.y = y

        def process_input(self, s: str) -> str:
            return s.replace(self.x, self.y)

        def process_output(self, s: str) -> str:
            return s.replace(self.y, self.x)

    def test_process_input(self) -> None:
        processor = input_parsing.CombinedProcessor([
            self.SubstitutionProcessor("a", "b"),
            self.SubstitutionProcessor("b", "e")
        ])
        result = processor.process_input("abca")
        expected = "eece"
        self.assertEquals(expected, result)

    def test_process_output(self) -> None:
        processor = input_parsing.CombinedProcessor([
            self.SubstitutionProcessor("a", "b"),
            self.SubstitutionProcessor("b", "e")
        ])
        result = processor.process_output("eece")
        expected = "aaca"
        self.assertEquals(expected, result)