import unittest
from typing import Tuple

import word_analysis


class VerbsTests(unittest.TestCase):

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

    def test_build_word_transformation_liegen_gelegen(self) -> None:
        subsequence_intervals = word_analysis.WordSubsequenceIntervals(word_analysis.LCSMatrix("liegen", "gelegen"))
        transf = word_analysis.build_word_transformation(subsequence_intervals)
        expected = word_analysis.WordTransformationSequence(
            [word_analysis.EditTransformation("ge", ""),
             word_analysis.SkipToTransformation('l', 'i'),
             word_analysis.EditTransformation('', "i"),
             word_analysis.SkipToTransformation("egen", "")]
        )
        self.assertEqual(transf, expected)
        transformed, transformee = transf.apply("", "liegen")
        self.assertEqual(transformed, "gelegen")
        self.assertEqual(transformee, "")

    def test_build_word_transformation_schmieren_geschmiert(self) -> None:
        subsequence_intervals = word_analysis.WordSubsequenceIntervals(word_analysis.LCSMatrix("schmieren", "geschmiert"))
        transf = word_analysis.build_word_transformation(subsequence_intervals)
        expected = word_analysis.WordTransformationSequence(
            [word_analysis.EditTransformation("ge", ""),
             word_analysis.SkipToTransformation("schmier", "en"),
             word_analysis.EditTransformation("t", "en")]
        )
        self.assertEqual(transf, expected)
        transformed, transformee = transf.apply("", "schmieren")
        self.assertEqual(transformed, "geschmiert")
        self.assertEqual(transformee, "")

    def test_common_prefix(self) -> None:
        self.assertEquals("foo", word_analysis.common_prefix("foobar", "foorab"))
        self.assertEquals("foo", word_analysis.common_prefix("foo", "foobar"))
        self.assertEquals("foo", word_analysis.common_prefix("foobar", "foo"))
        self.assertEquals("", word_analysis.common_prefix("foobar", "herbert"))

    def test_common_suffix(self) -> None:
        self.assertEquals("bar", word_analysis.common_suffix("foobar", "oofbar"))
        self.assertEquals("bar", word_analysis.common_suffix("foobar", "bar"))
        self.assertEquals("bar", word_analysis.common_suffix("bar", "foobar"))
        self.assertEquals("", word_analysis.common_suffix("foobar", "herbert"))


class LCSMatrixTests(unittest.TestCase):

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


class EditTransformationTests(unittest.TestCase):

    def test_apply_insert(self) -> None:
        transf = word_analysis.EditTransformation("bar", "")
        transformed, transformee = transf.apply("foo", "2")
        self.assertEqual(transformed, "foobar")
        self.assertEqual(transformee, "2")

    def test_apply_match(self) -> None:
        transf = word_analysis.EditTransformation("ugo", "il")
        transformed, transformee = transf.apply("h", "ilbert")
        self.assertEqual(transformed, "hugo")
        self.assertEqual(transformee, "bert")

    def test_apply_delete(self) -> None:
        transf = word_analysis.EditTransformation('', "ba")
        transformed, transformee = transf.apply("foo", "bar2")
        self.assertEqual(transformed, "foo")
        self.assertEqual(transformee, "r2")

    def test_apply_unmatch_(self) -> None:
        transf = word_analysis.EditTransformation("hugo", "egon")
        self.assertRaises(ValueError, transf.apply, "alf", "hugabo")

    def test_maybe_joinable(self) -> None:
        transf1 = word_analysis.EditTransformation("ugo", "il")
        transf2 = word_analysis.EditTransformation("ugo", "il")
        transf3 = word_analysis.EditTransformation("ugo", "")
        transf4 = word_analysis.EditTransformation("", "il")
        self.assertTrue(transf1.maybe_joinable(transf2))
        self.assertTrue(transf2.maybe_joinable(transf1))
        self.assertFalse(transf1.maybe_joinable(transf3))
        self.assertFalse(transf3.maybe_joinable(transf1))
        self.assertFalse(transf1.maybe_joinable(transf4))
        self.assertFalse(transf4.maybe_joinable(transf1))
        self.assertFalse(transf2.maybe_joinable(transf3))

    def test_equals(self) -> None:
        transf1 = word_analysis.EditTransformation("ugo", "il")
        transf2 = word_analysis.EditTransformation("ugo", "il")
        transf3 = word_analysis.EditTransformation("ugo", "")
        transf4 = word_analysis.EditTransformation("", "il")
        self.assertEquals(transf1, transf2)
        self.assertEquals(transf2 ,transf1)
        self.assertNotEquals(transf1, transf3)
        self.assertNotEquals(transf3, transf1)
        self.assertNotEquals(transf1, transf4)
        self.assertNotEquals(transf4, transf1)
        self.assertNotEquals(transf2, transf3)

    def test_hash(self) -> None:
        raise NotImplementedError()


class SkipToTransformationTests(unittest.TestCase):

    def test_apply_unmatch(self) -> None:
        transf = word_analysis.SkipToTransformation("foo", "bar")
        self.assertRaises(ValueError, transf.apply, "pete", "fobare")
        self.assertRaises(ValueError, transf.apply, "pete", "foobae")
        self.assertRaises(ValueError, transf.apply, "pete", "fo0bae")

    def test_apply_no_jump(self) -> None:
        transf = word_analysis.SkipToTransformation("ob", "a")
        transformed, transformee = transf.apply("f", "obarfoobar")
        self.assertEqual(transformed, "fob")
        self.assertEqual(transformee, "arfoobar")

    def test_apply_with_jump(self) -> None:
        transf = word_analysis.SkipToTransformation("ob", "a")
        transformed, transformee = transf.apply("f", "ooboobarfoobar")
        self.assertEqual(transformed, "fooboob")
        self.assertEqual(transformee, "arfoobar")

    def test_maybe_joinable(self) -> None:
        transf1 = word_analysis.SkipToTransformation("foo", "bar")
        transf2 = word_analysis.SkipToTransformation("abc", "def")
        self.assertTrue(transf1.maybe_joinable(transf2))
        self.assertTrue(transf2.maybe_joinable(transf2))
        self.assertFalse(transf1.maybe_joinable(word_analysis.EditTransformation("hugo", "herbert")))

    def test_equals(self) -> None:
        transf1 = word_analysis.SkipToTransformation("foo", "bar")
        transf11 = word_analysis.SkipToTransformation("foo", "bar")
        transf2 = word_analysis.SkipToTransformation("abc", "def")
        self.assertEquals(transf1, transf1)
        self.assertEquals(transf1, transf11)
        self.assertNotEquals(transf1, transf2)
        self.assertNotEquals(transf1, word_analysis.EditTransformation("hugo", "herbert"))

    def test_hash(self) -> None:
        raise NotImplementedError()


class TransformationSequenceTests(unittest.TestCase):

    def test_apply(self) -> None:
        transf = word_analysis.WordTransformationSequence(
            [word_analysis.SkipToTransformation("f", "uncti"),
             word_analysis.EditTransformation('', "uncti"),
             word_analysis.SkipToTransformation("o", "n"),
             word_analysis.EditTransformation("o", "n"),
             word_analysis.EditTransformation("bar", "")]
        )
        transformed, transformee = transf.apply("", "function")
        self.assertEqual(transformed, "foobar")
        self.assertEqual(transformee, "")

    def test_maybe_joinable(self) -> None:
        raise NotImplementedError()

    def test_equals(self) -> None:
        raise NotImplementedError()

    def test_hash(self) -> None:
        raise NotImplementedError()