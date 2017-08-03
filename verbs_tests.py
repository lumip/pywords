import unittest

import verbs


class VerbsTests(unittest.TestCase):

    def __lcs_matrix_test_worker(self, word_a: str, word_b: str, expected: verbs.Matrix) -> None:
        lcs = verbs.LCSMatrix(word_a, word_b)
        self.assertEqual(lcs.word_a, word_a)
        self.assertEqual(lcs.word_b, word_b)
        self.assertEqual(lcs.matrix, expected)
        self.assertEqual(lcs.edit_distance, expected[-1][-1])

    def test_build_lcs_matrix_liegen_gelegen(self):
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

    def test_build_lcs_matrix_hallo_hello(self):
        word_a = "hallo"
        word_b = "hello"
        expected = ((0, 1, 2, 3, 4, 5),
                    (1, 0, 1, 2, 3, 4),
                    (2, 1, 1, 2, 3, 4),
                    (3, 2, 2, 1, 2, 3),
                    (4, 3, 3, 2, 1, 2),
                    (5, 4, 4, 3, 2, 1))
        self.__lcs_matrix_test_worker(word_a, word_b, expected)

    def test_build_lcs_matrix_asdf_asdf(self):
        word_a = "asdf"
        word_b = "asdf"
        expected = ((0, 1, 2, 3, 4),
                    (1, 0, 1, 2, 3),
                    (2, 1, 0, 1, 2),
                    (3, 2, 1, 0, 1),
                    (4, 3, 2, 1, 0))
        self.__lcs_matrix_test_worker(word_a, word_b, expected)

    def test_build_lcs_matrix_fasd_asdf(self):
        word_a = "fasd"
        word_b = "asdf"
        expected = ((0, 1, 2, 3, 4),
                    (1, 1, 2, 3, 3),
                    (2, 1, 2, 3, 4),
                    (3, 2, 1, 2, 3),
                    (4, 3, 2, 1, 2))
        self.__lcs_matrix_test_worker(word_a, word_b, expected)

    def test_build_lcs_matrix_halloh_hello(self):
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

    def test_build_intervals_liegen_gelegen(self):
        interval_pairs = verbs.get_common_subsequence_intervals(verbs.LCSMatrix("liegen", "gelegen"))
        #expected_interval_pairs = [verbs.IntervalPair(verbs.Interval(0, 2), verbs.Interval(0, 3), False),
        #                           verbs.IntervalPair(verbs.Interval(2, 6), verbs.Interval(3, 7), True)]
        # todo: is that a desirable output? probably not -> the above result is achieved with prioritizing edits or deletes over inserts
        # the lower result is achieved with prioritizing inserts in draws. is this always desirable? instead
        # of establishing an order, should the costs for steps in the lcs computation be adjusted?
        expected_interval_pairs = [verbs.IntervalPair(verbs.Interval(0, 0), verbs.Interval(0, 2), False),
                                   verbs.IntervalPair(verbs.Interval(0, 1), verbs.Interval(2, 3), True),
                                   verbs.IntervalPair(verbs.Interval(1, 2), verbs.Interval(3, 3), False),
                                   verbs.IntervalPair(verbs.Interval(2, 6), verbs.Interval(3, 7), True)]
        self.assertEqual(interval_pairs, expected_interval_pairs)

    def test_build_intervals_halloh_hello(self):
        interval_pairs = verbs.get_common_subsequence_intervals(verbs.LCSMatrix("halloh", "hello"))
        expected_interval_pairs = [verbs.IntervalPair(verbs.Interval(0, 1), verbs.Interval(0, 1), True),
                                   verbs.IntervalPair(verbs.Interval(1, 2), verbs.Interval(1, 2), False),
                                   verbs.IntervalPair(verbs.Interval(2, 5), verbs.Interval(2, 5), True),
                                   verbs.IntervalPair(verbs.Interval(5, 6), verbs.Interval(5, 5), False)]
        self.assertEqual(interval_pairs, expected_interval_pairs)