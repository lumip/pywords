import unittest
from typing import Tuple

import word_analysis


class VerbsTests(unittest.TestCase):

    def __lcs_matrix_test_worker(self, word_a: str, word_b: str, expected: word_analysis.Matrix) -> None:
        lcs = word_analysis.LCSMatrix(word_a, word_b)
        self.assertEqual(lcs.word_a, word_a)
        self.assertEqual(lcs.word_b, word_b)
        self.assertEqual(lcs.matrix, expected)
        self.assertEqual(lcs.edit_distance, expected[-1][-1])

    def test_build_lcs_matrix_liegen_gelegen(self) -> None:
        word_a = "liegen"
        word_b = "gelegen"
        expected = ((0, 1, 2, 3, 4, 5, 6, 7),
                    (1, 1, 2, 2, 3, 4, 5, 6),
                    (2, 2, 2, 3, 3, 4, 5, 6),
                    (3, 3, 2, 3, 3, 4, 4, 5),
                    (4, 3, 3, 3, 4, 3, 4, 5),
                    (5, 4, 3, 4, 3, 4, 3, 4),
                    (6, 5, 4, 4, 4, 4, 4, 3))
        self.__lcs_matrix_test_worker(word_a, word_b, expected)

    def test_build_lcs_matrix_hallo_hello(self) -> None:
        word_a = "hallo"
        word_b = "hello"
        expected = ((0, 1, 2, 3, 4, 5),
                    (1, 0, 1, 2, 3, 4),
                    (2, 1, 1, 2, 3, 4),
                    (3, 2, 2, 1, 2, 3),
                    (4, 3, 3, 2, 1, 2),
                    (5, 4, 4, 3, 2, 1))
        self.__lcs_matrix_test_worker(word_a, word_b, expected)

    def test_build_lcs_matrix_asdf_asdf(self) -> None:
        word_a = "asdf"
        word_b = "asdf"
        expected = ((0, 1, 2, 3, 4),
                    (1, 0, 1, 2, 3),
                    (2, 1, 0, 1, 2),
                    (3, 2, 1, 0, 1),
                    (4, 3, 2, 1, 0))
        self.__lcs_matrix_test_worker(word_a, word_b, expected)

    def test_build_lcs_matrix_fasd_asdf(self) -> None:
        word_a = "fasd"
        word_b = "asdf"
        expected = ((0, 1, 2, 3, 4),
                    (1, 1, 2, 3, 3),
                    (2, 1, 2, 3, 4),
                    (3, 2, 1, 2, 3),
                    (4, 3, 2, 1, 2))
        self.__lcs_matrix_test_worker(word_a, word_b, expected)

    def test_build_lcs_matrix_halloh_hello(self) -> None:
        word_a = "halloh"
        word_b = "hello"
        expected = ((0, 1, 2, 3, 4, 5),
                    (1, 0, 1, 2, 3, 4),
                    (2, 1, 1, 2, 3, 4),
                    (3, 2, 2, 1, 2, 3),
                    (4, 3, 3, 2, 1, 2),
                    (5, 4, 4, 3, 2, 1),
                    (6, 5, 5, 4, 3, 2))
        self.__lcs_matrix_test_worker(word_a, word_b, expected)

    def __lcs_intervals_test_worker(self, word_a: str, word_b: str, expected: Tuple[word_analysis.IntervalPair]) -> None:
        intvs = word_analysis.WordSubsequenceIntervals(word_analysis.LCSMatrix(word_a, word_b))
        self.assertEqual(intvs.word_a, word_a)
        self.assertEqual(intvs.word_b, word_b)
        self.assertEqual(intvs.intervals, expected)

    def test_build_intervals_liegen_gelegen(self) -> None:
        word_a = "liegen"
        word_b = "gelegen"
        #expected_interval_pairs = [verbs.IntervalPair(verbs.Interval(0, 2), verbs.Interval(0, 3), False),
        #                           verbs.IntervalPair(verbs.Interval(2, 6), verbs.Interval(3, 7), True)]
        # todo: is that a desirable output? probably not -> the above result is achieved with prioritizing edits or deletes over inserts
        # the lower result is achieved with prioritizing inserts in draws. is this always desirable? instead
        # of establishing an order, should the costs for steps in the lcs computation be adjusted?
        expected_interval_pairs = (word_analysis.IntervalPair(word_analysis.Interval(0, 0), word_analysis.Interval(0, 2), False),
                                   word_analysis.IntervalPair(word_analysis.Interval(0, 1), word_analysis.Interval(2, 3), True),
                                   word_analysis.IntervalPair(word_analysis.Interval(1, 2), word_analysis.Interval(3, 3), False),
                                   word_analysis.IntervalPair(word_analysis.Interval(2, 6), word_analysis.Interval(3, 7), True))
        self.__lcs_intervals_test_worker(word_a, word_b, expected_interval_pairs)

    def test_build_intervals_halloh_hello(self) -> None:
        word_a = "halloh"
        word_b = "hello"
        expected_interval_pairs = (word_analysis.IntervalPair(word_analysis.Interval(0, 1), word_analysis.Interval(0, 1), True),
                                   word_analysis.IntervalPair(word_analysis.Interval(1, 2), word_analysis.Interval(1, 2), False),
                                   word_analysis.IntervalPair(word_analysis.Interval(2, 5), word_analysis.Interval(2, 5), True),
                                   word_analysis.IntervalPair(word_analysis.Interval(5, 6), word_analysis.Interval(5, 5), False))
        self.__lcs_intervals_test_worker(word_a, word_b, expected_interval_pairs)

    def test_insert_transformation(self) -> None:
        transf = word_analysis.InsertTransformation("bar")
        transformed, transformee = transf.apply("foo", "2")
        self.assertEqual(transformed, "foobar")
        self.assertEqual(transformee, "2")

    def test_edit_transformation(self) -> None:
        transf = word_analysis.EditTransformation("ugo")
        transformed, transformee = transf.apply("h", "ilbert")
        self.assertEqual(transformed, "hugo")
        self.assertEqual(transformee, "ert")

    def test_delete_transformation(self) -> None:
        transf = word_analysis.DeleteTransformation(2)
        transformed, transformee = transf.apply("foo", "bar2")
        self.assertEqual(transformed, "foo")
        self.assertEqual(transformee, "r2")

    def test_skip_transformation(self) -> None:
        transf = word_analysis.SkipTransformation(2)
        transformed, transformee = transf.apply("foo", "bar2")
        self.assertEqual(transformed, "fooba")
        self.assertEqual(transformee, "r2")

    def test_transformation_sequence(self) -> None:
        transf = word_analysis.WordTransformationSequence(
            [word_analysis.SkipTransformation(1),
             word_analysis.DeleteTransformation(5),
             word_analysis.SkipTransformation(1),
             word_analysis.EditTransformation("o"),
             word_analysis.InsertTransformation("bar")]
        )
        transformed, transformee = transf.apply("", "function")
        self.assertEqual(transformed, "foobar")
        self.assertEqual(transformee, "")

    def test_build_word_transformation(self) -> None:
        subsequence_intervals = word_analysis.WordSubsequenceIntervals(word_analysis.LCSMatrix("liegen", "gelegen"))
        transf = word_analysis.build_word_transformation(subsequence_intervals)
        expected = word_analysis.WordTransformationSequence(
            [word_analysis.InsertTransformation("ge"),
             word_analysis.SkipTransformation(1),
             word_analysis.DeleteTransformation(1),
             word_analysis.SkipTransformation(4)]
        )
        self.assertEqual(transf, expected)
        transformed, transformee = transf.apply("", "liegen")
        self.assertEqual(transformed, "gelegen")
        self.assertEqual(transformee, "")
