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

import abc
from typing import List


class WordProcessor(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def process_input(self, s: str) -> str:
        pass

    @abc.abstractmethod
    def process_output(self, s: str) -> str:
        pass


class HangeulComposer(WordProcessor):

    # implemented according to http://unicode.org/versions/Unicode5.0.0/ch03.pdf#G24646

    def __init__(self) -> None:
        self.S_BASE = 0xAC00
        self.L_BASE = 0x1100
        self.V_BASE = 0x1161
        self.T_BASE = 0x11A7
        self.L_COUNT = 19
        self.V_COUNT = 21
        self.T_COUNT = 28
        self.N_COUNT = self.V_COUNT * self.T_COUNT
        self.S_COUNT = self.N_COUNT * self.L_COUNT

    def __is_out_of_bounds(self, i: int, max: int) -> bool:
        return i < 0 or i >= max

    def decompose(self, input: str) -> str:
        output = ""
        for c in input:
            s_index = ord(c) - self.S_BASE
            if self.__is_out_of_bounds(s_index, self.S_COUNT):
                output += c
                continue
            l_index = self.L_BASE + (s_index // self.N_COUNT)
            v_index = self.V_BASE + ((s_index % self.N_COUNT) // self.T_COUNT)
            t_index = self.T_BASE + (s_index % self.T_COUNT)
            output += chr(l_index)
            output += chr(v_index)
            if t_index > self.T_BASE:
                output += chr(t_index)
        return output

    def compose(self, input: str) -> str:
        output = ""
        bases = [self.L_BASE, self.V_BASE, self.T_BASE]
        counts = [self.L_COUNT, self.V_COUNT, self.T_COUNT]
        i = 0
        while i < len(input):
            j = 0
            syllable_inds = [0, 0, 0]
            while j < 3 and i + j < len(input):
                syllable_inds[j] = ord(input[i + j]) - bases[j]
                j += 1
            if (self.__is_out_of_bounds(syllable_inds[0], counts[0]) or
                self.__is_out_of_bounds(syllable_inds[1], counts[1])):
                output += input[i + 0]
                i += 1
                continue
            if self.__is_out_of_bounds(syllable_inds[2], counts[2]):
                syllable_inds[2] = 0
                j = 2
            composed = (syllable_inds[0] * self.V_COUNT + syllable_inds[1]) * self.T_COUNT + syllable_inds[2] + self.S_BASE
            output += chr(composed)
            i += j
        return output

    def process_input(self, s: str) -> str:
        return self.decompose(s)

    def process_output(self, s: str) -> str:
        return self.compose(s)


class StripProcessor(WordProcessor):

    def process_input(self, s: str) -> str:
        return s.strip()

    def process_output(self, s: str) -> str:
        return s


class CombinedProcessor(WordProcessor):

    def __init__(self, processors: List[WordProcessor]):
        self.__processors = processors.copy()

    def process_input(self, s: str) -> str:
        for proc in self.__processors:
            s = proc.process_input(s)
        return s

    def process_output(self, s: str) -> str:
        for proc in reversed(self.__processors):
            s = proc.process_output(s)
        return s