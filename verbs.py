import sys
import codecs
from typing import List, Tuple, NamedTuple, TypeVar
import abc

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
        if start > end:
            raise ValueError("Cannot declare Interval with start after end")
        self.__start = start
        self.__end = end

    @property
    def start(self) -> int:
        return self.__start

    @property
    def end(self) -> int:
        return self.__end

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def empty(self) -> bool:
        return self.length <= 0

    def __eq__(self, other) -> bool:
        if not isinstance(other, Interval):
            return False
        return other.start == self.start and other.end == self.end

    def __str__(self) -> str:
        return "[" + str(self.start) + ", " + str(self.end) + "]"

    def __repr__(self) -> str:
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, IntervalPair):
            return False
        return other.interval_a == self.interval_a and other.interval_b == self.interval_b and other.common == self.common

    def __str__(self) -> str:
        return "(" + str(self.interval_a) + " : " + str(self.interval_b) + " : " + str(self.common) + ")"

    def __repr__(self) -> str:
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
    def matrix(self) -> Matrix:
        return self.__matrix

    @property
    def edit_distance(self) -> int:
        return self.matrix[-1][-1]


class WordSubsequenceIntervals:

    def __init__(self, word_pair_lcs_matrix: LCSMatrix) -> None:
        self.__word_a = word_pair_lcs_matrix.word_a
        self.__word_b = word_pair_lcs_matrix.word_b
        self.__intervals = self.__get_common_subsequence_intervals(word_pair_lcs_matrix)

    @staticmethod
    def __get_common_subsequence_intervals(word_pair_lcs_matrix: LCSMatrix) -> Tuple[IntervalPair]:
        word_a = word_pair_lcs_matrix.word_a
        word_b = word_pair_lcs_matrix.word_b
        lcs_matrix = word_pair_lcs_matrix.matrix
        intervals = []
        interval_pair_builder = IntervalPairBuilder() # allows us to conveniently keep track of last interval borders and build intervals sequentially
        interval_pair_builder.set_end_a(len(word_a))
        interval_pair_builder.set_end_b(len(word_b))
        last_letter_common = word_a[-1] == word_b[-1]
        if last_letter_common:
            interval_pair_builder.set_common()
        i, j = len(word_a), len(word_b)
        while i > 0 and j > 0:
            old_i = i
            old_j = j
            current_letter_common = (word_a[i - 1] == word_b[j - 1])
            if current_letter_common:
                i = i - 1
                j = j - 1
            else:
                # todo: prioritizing delete steps in draws might give more favorable results? but always?
                # it splits "liegen" and "gelegen" into |  |l|i|egen|
                #                                       |ge|l| |egen|
                # instead of | li|egen|
                #            |gel|egen|
                step, _ = indmin([lcs_matrix[i - 1][j], lcs_matrix[i - 1][j - 1], lcs_matrix[i][j - 1]])
                assert (step in range(0, 4))
                if step == 0:  # delete step
                    i = i - 1
                elif step == 1:  # edit step
                    i = i - 1
                    j = j - 1
                elif step == 2:  # insert step
                    j = j - 1
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
        return tuple(reversed(intervals))

    @property
    def word_a(self) -> str:
        return self.__word_a

    @property
    def word_b(self) -> str:
        return self.__word_b

    @property
    def intervals(self) -> Tuple[IntervalPair]:
        return self.__intervals

    def get_subsequences(self, interval_pair: IntervalPair) -> Tuple[str, str]:
        return (self.word_a[interval_pair.interval_a.start:interval_pair.interval_a.end],
                self.word_b[interval_pair.interval_b.start:interval_pair.interval_b.end])


class WordTransformation(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @staticmethod
    def check_length_raise_error(transformee: str, number_of_chars: int) -> None:
        if len(transformee) < number_of_chars:
            ValueError("Transformee <" + transformee + "> is too short")

    @staticmethod
    def snip_transformee(transformee: str, number_of_chars: int) -> str:
        WordTransformation.check_length_raise_error(transformee, number_of_chars)
        return transformee[number_of_chars:]

    @abc.abstractmethod
    def apply(self, transformed: str, transformee: str) -> Tuple[str, str]:
        pass

class InsertTransformation(WordTransformation):

    def __init__(self, insertee: str) -> None:
        self.__insertee = insertee

    def apply(self, transformed: str, transformee: str) -> Tuple[str, str]:
        return (transformed + self.__insertee), transformee

    def __eq__(self, other) -> bool:
        if not isinstance(other, InsertTransformation): return False
        return other.__insertee == self.__insertee

class EditTransformation(WordTransformation):

    def __init__(self, insertee: str) -> None:
        self.__insertee = insertee

    def apply(self, transformed: str, transformee: str) -> Tuple[str, str]:
        self.check_length_raise_error(transformee, len(self.__insertee))
        return (transformed + self.__insertee), self.snip_transformee(transformee, len(self.__insertee))

    def __eq__(self, other) -> bool:
        if not isinstance(other, EditTransformation): return False
        return other.__insertee == self.__insertee

class DeleteTransformation(WordTransformation):

    def __init__(self, number_of_chars: int) -> None:
        self.__number_of_chars = number_of_chars

    def apply(self, transformed: str, transformee: str) -> Tuple[str, str]:
        self.check_length_raise_error(transformee, self.__number_of_chars)
        return transformed, self.snip_transformee(transformee, self.__number_of_chars)

    def __eq__(self, other) -> bool:
        if not isinstance(other, DeleteTransformation): return False
        return other.__number_of_chars == self.__number_of_chars

class SkipTransformation(WordTransformation):

    def __init__(self, number_of_chars: int) -> None:
        self.__number_of_chars = number_of_chars

    def apply(self, transformed: str, transformee: str) -> Tuple[str, str]:
        self.check_length_raise_error(transformee, self.__number_of_chars)
        return (transformed + transformee[:self.__number_of_chars]), self.snip_transformee(transformee, self.__number_of_chars)

    def __eq__(self, other) -> bool:
        if not isinstance(other, SkipTransformation): return False
        return other.__number_of_chars == self.__number_of_chars

class WordTransformationSequence(WordTransformation):

    def __init__(self, transformations: List[WordTransformation]) -> None:
        self.__transformations = tuple(transformations)

    def apply(self, transformed: str, transformee: str) -> Tuple[str, str]:
        for transformation in self.__transformations:
            transformed, transformee = transformation.apply(transformed, transformee)
        return transformed, transformee

    def __eq__(self, other) -> bool:
        if not isinstance(other, WordTransformationSequence): return False
        return other.__transformations == self.__transformations

def build_word_transformation(subsequence_intervals: WordSubsequenceIntervals) -> WordTransformation:
    transforms = []
    for interval_pair in subsequence_intervals.intervals:
        transform = None
        subsequence_a, subsequence_b = subsequence_intervals.get_subsequences(interval_pair)
        if interval_pair.common:
            transform = SkipTransformation(interval_pair.interval_a.length)
        elif interval_pair.interval_a.empty:
            transform = InsertTransformation(subsequence_b)
        elif interval_pair.interval_b.empty:
            transform = DeleteTransformation(interval_pair.interval_a.length)
        else:
            transform = EditTransformation(subsequence_b)
        transforms.append(transform)
    return WordTransformationSequence(transforms)