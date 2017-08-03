import sys
import codecs
from typing import List, Tuple, NamedTuple, TypeVar
from collections import namedtuple

#training_set = []
#with codecs.open("words.txt", 'r', encoding='utf-8') as f:
#    for line in f.readlines():
#        training_set.append([part.strip() for part in line.split(",")])
#
#test = ["liegen", "gelegen"]

MutableMatrix = List[List[int]]
Matrix = Tuple[Tuple[int]]

def make_matrix_immutable(mat: MutableMatrix) -> Matrix:
    return tuple(tuple(row) for row in mat)

def make_matrix_mutable(mat: Matrix) -> MutableMatrix:
    return list(list(row) for row in mat)

T = TypeVar('T')

def indmin(values: List[T]) -> Tuple[int, T]:
    return min(enumerate(values), key=lambda p: p[1])


class Interval:

    def __init__(self, start: int, end: int) -> None:
        self.__start = start
        self.__end = end

    @property
    def start(self) -> int:
        return self.__start

    @property
    def end(self) -> int:
        return self.__end

    def __eq__(self, other):
        if not isinstance(other, Interval):
            return False
        return other.start == self.start and other.end == self.end

    def __str__(self):
        return "[" + str(self.start) + ", " + str(self.end) + "]"

    def __repr__(self):
        return str(self) + "@" + str(id(self))


class IntervalPair:

    def __init__(self, interval_a: Interval, interval_b: Interval, common: bool):
        self.__interval_a = interval_a
        self.__interval_b = interval_b
        self.__common = common

    @property
    def interval_a(self) -> Interval:
        return self.__interval_a

    @property
    def interval_b(self) -> Interval:
        return self.__interval_b

    @property
    def common(self) -> bool:
        return self.__common

    def __eq__(self, other):
        if not isinstance(other, IntervalPair):
            return False
        return other.interval_a == self.interval_a and other.interval_b == self.interval_b and other.common == self.common

    def __str__(self):
        return "(" + str(self.interval_a) + " : " + str(self.interval_b) + " : " + str(self.common) + ")"

    def __repr__(self):
        return str(self) + "@" + str(id(self))


class IntervalPairBuilder:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.interval_a = Interval(0, 0)
        self.interval_b = Interval(0, 0)
        self.common = False

    def prepare_next(self) -> None:
        self.interval_a = Interval(0, self.interval_a.start)
        self.interval_b = Interval(0, self.interval_b.start)
        self.common = not self.common

    def set_start_a(self, start: int) -> None:
        self.interval_a = Interval(start, self.interval_a.end)

    def set_end_a(self, end: int) -> None:
        self.interval_a = Interval(self.interval_a.start, end)

    def set_start_b(self, start: int) -> None:
        self.interval_b = Interval(start, self.interval_b.end)

    def set_end_b(self, end: int) -> None:
        self.interval_b = Interval(self.interval_b.start, end)

    def set_common(self) -> None:
        self.common = True

    def build(self) -> IntervalPair:
        return IntervalPair(self.interval_a, self.interval_b, self.common)


class LCSMatrix:

    def __init__(self, word_a: str, word_b: str) -> None:
        self.__word_a = word_a
        self.__word_b = word_b
        self.__matrix = self.__compute_lcs_matrix(self.__word_a, self.__word_b)

    @staticmethod
    def __compute_lcs_matrix(word_a: str, word_b: str) -> Matrix:
        edit_cost, insert_cost, delete_cost = 1, 1, 1  # the perspective is: change word a into b
        cols = len(word_b) + 1
        rows = len(word_a) + 1
        lcs = [[i + j for j in range(cols)] for i in range(rows)]  # type: MutableMatrix
        for i in range(1, rows):
            for j in range(1, cols):
                step_cost = 0
                if word_a[i - 1] != word_b[j - 1]:
                    step_cost = edit_cost
                d_edit = lcs[i - 1][j - 1] + step_cost
                d_delete = lcs[i - 1][j] + delete_cost
                d_insert = lcs[i][j - 1] + insert_cost
                lcs[i][j] = min(d_edit, d_insert, d_delete)
        return tuple(tuple(row) for row in lcs)

    @property
    def word_a(self) -> str:
        return self.__word_a

    @property
    def word_b(self) -> str:
        return self.__word_b

    @property
    def matrix(self):
        return self.__matrix

    @property
    def edit_distance(self):
        return self.matrix[-1][-1]


def get_common_subsequence_intervals(word_pair_lcs_matrix: LCSMatrix) -> List[IntervalPair]:
    word_a = word_pair_lcs_matrix.word_a
    word_b = word_pair_lcs_matrix.word_b
    lcs_matrix = word_pair_lcs_matrix.matrix
    intervals = []
    interval_pair_builder = IntervalPairBuilder()
    interval_pair_builder.set_end_a(len(word_a))
    interval_pair_builder.set_end_b(len(word_b))
    last_letter_common = word_a[-1] == word_b[-1]
    if last_letter_common:
        interval_pair_builder.set_common()
    i, j = len(word_a), len(word_b)
    while i > 0 and j > 0:
        old_i = i
        old_j = j
        current_letter_common = word_a[i-1] == word_b[j-1]
        if current_letter_common:
            i = i-1
            j = j-1
        else:
            # todo: prioritizing delete steps in draws might give more favorable results? but always?
            # it splits "liegen" and "gelegen" into |  |l|i|egen|
            #                                       |ge|l| |egen|
            # instead of | li|egen|
            #            |gel|egen|
            step, _ = indmin([lcs_matrix[i-1][j], lcs_matrix[i - 1][j - 1], lcs_matrix[i][j-1]])
            assert(step in range(0, 4))
            if step == 0:  # delete step
                i = i-1
            elif step == 1:  # edit step
                i = i-1
                j = j-1
            elif step == 2:  # insert step
                j = j-1
        if current_letter_common != last_letter_common:
            interval_pair_builder.set_start_a(old_i)
            interval_pair_builder.set_start_b(old_j)
            intervals.append(interval_pair_builder.build())
            interval_pair_builder.prepare_next()
        last_letter_common = current_letter_common
    interval_pair_builder.set_start_a(i)
    interval_pair_builder.set_start_b(j)
    intervals.append(interval_pair_builder.build())
    if (i > 0 or j > 0) and last_letter_common:
        interval_pair_builder.prepare_next()
        intervals.append(interval_pair_builder.build())
    return list(reversed(intervals))
