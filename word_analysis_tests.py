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
            [word_analysis.EditTransformation("", "", "ge"),
             word_analysis.EditTransformation("l", "i", ""),
             word_analysis.EditTransformation("egen", "", "")]
        )
        self.assertEqual(transf, expected)
        transformed, transformee = transf.apply_step("", "liegen")
        self.assertEqual(transformed, "gelegen")
        self.assertEqual(transformee, "")

    def test_build_word_transformation_schmieren_geschmiert(self) -> None:
        subsequence_intervals = word_analysis.WordSubsequenceIntervals(word_analysis.LCSMatrix("schmieren", "geschmiert"))
        transf = word_analysis.build_word_transformation(subsequence_intervals)
        expected = word_analysis.WordTransformationSequence(
            [word_analysis.EditTransformation("", "", "ge"),
             #word_analysis.SkipToTransformation("schmier", "en"),
             word_analysis.EditTransformation("schmier", "en", "t")]
        )
        self.assertEqual(transf, expected)
        transformed, transformee = transf.apply_step("", "schmieren")
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
        transf = word_analysis.EditTransformation("oo", "", "bar")
        transformed, transformee = transf.apply_step("f", "oo2")
        self.assertEqual(transformed, "foobar")
        self.assertEqual(transformee, "2")

    def test_apply_match(self) -> None:
        transf = word_analysis.EditTransformation("u", "il", "go")
        transformed, transformee = transf.apply_step("h", "uilbert")
        self.assertEqual(transformed, "hugo")
        self.assertEqual(transformee, "bert")

    def test_apply_delete(self) -> None:
        transf = word_analysis.EditTransformation("", "ba", "")
        transformed, transformee = transf.apply_step("foo", "bar2")
        self.assertEqual(transformed, "foo")
        self.assertEqual(transformee, "r2")

    def test_apply_unmatch_(self) -> None:
        transf1 = word_analysis.EditTransformation("", "egon", "hugo")
        self.assertRaises(ValueError, transf1.apply_step, "alf", "hugabo")
        transf2 = word_analysis.EditTransformation("egon", "", "hugo")
        self.assertRaises(ValueError, transf2.apply_step, "alf", "hugabo")
        transf3 = word_analysis.EditTransformation("eg", "on", "hugo")
        self.assertRaises(ValueError, transf3.apply_step, "alf", "hugabo")

    def test_maybe_joinable(self) -> None:
        transf1 = word_analysis.EditTransformation("", "ugo", "il")
        transf2 = word_analysis.EditTransformation("", "ugo", "il")
        transf3 = word_analysis.EditTransformation("", "ugo", "")
        transf4 = word_analysis.EditTransformation("", "", "il")
        transf5 = word_analysis.EditTransformation("foo", "ugo", "il")
        transf6 = word_analysis.EditTransformation("doo", "ugo", "il")
        transf7 = word_analysis.EditTransformation("oo", "ug", "il")
        self.assertTrue(transf1.maybe_joinable(transf1))
        self.assertTrue(transf2.maybe_joinable(transf2))
        self.assertTrue(transf3.maybe_joinable(transf3))
        self.assertTrue(transf4.maybe_joinable(transf4))
        self.assertTrue(transf1.maybe_joinable(transf2))
        self.assertTrue(transf2.maybe_joinable(transf1))
        self.assertTrue(transf5.maybe_joinable(transf6))
        self.assertTrue(transf6.maybe_joinable(transf5))
        self.assertTrue(transf5.maybe_joinable(transf1))
        self.assertTrue(transf1.maybe_joinable(transf5))
        self.assertFalse(transf1.maybe_joinable(transf3))
        self.assertFalse(transf3.maybe_joinable(transf1))
        self.assertFalse(transf1.maybe_joinable(transf4))
        self.assertFalse(transf4.maybe_joinable(transf1))
        self.assertFalse(transf2.maybe_joinable(transf3))
        self.assertFalse(transf7.maybe_joinable(transf6))
        self.assertFalse(transf6.maybe_joinable(transf7))
        self.assertFalse(transf1.maybe_joinable(word_analysis.WordTransformationSequence([transf1])))

    def test_equals(self) -> None:
        transf1 = word_analysis.EditTransformation("", "ugo", "il")
        transf2 = word_analysis.EditTransformation("", "ugo", "il")
        transf3 = word_analysis.EditTransformation("", "ugo", "")
        transf4 = word_analysis.EditTransformation("", "", "il")
        transf5 = word_analysis.EditTransformation("foo", "ugo", "il")
        transf5b = word_analysis.EditTransformation("foo", "ugo", "il")
        transf6 = word_analysis.EditTransformation("oo", "ugo", "il")
        transf7 = word_analysis.EditTransformation("oo", "ug", "il")
        self.assertEquals(transf1, transf2)
        self.assertEquals(transf2 ,transf1)
        self.assertEquals(transf5, transf5b)
        self.assertEquals(transf5b, transf5)
        self.assertNotEquals(transf1, transf3)
        self.assertNotEquals(transf3, transf1)
        self.assertNotEquals(transf1, transf4)
        self.assertNotEquals(transf4, transf1)
        self.assertNotEquals(transf2, transf3)
        self.assertNotEquals(transf5, transf1)
        self.assertNotEquals(transf1, transf5)
        self.assertNotEquals(transf5, transf6)
        self.assertNotEquals(transf6, transf5)
        self.assertNotEquals(transf6, transf7)
        self.assertNotEquals(transf7, transf6)
        self.assertNotEquals(transf1, word_analysis.SkipToTransformation("ugo","il"))
        self.assertNotEquals(transf1, word_analysis.WordTransformationSequence([transf1]))

    def test_join_unjoinable(self) -> None:
        transf1 = word_analysis.EditTransformation("oo", "ugo", "il")
        transf3 = word_analysis.EditTransformation("oo", "ugo", "")
        transf4 = word_analysis.EditTransformation("oo", "", "il")

        with self.assertRaises(ValueError):
            transf1.join(transf3)
            transf1.join(transf4)
            transf3.join(transf1)
            transf4.join(transf1)
            transf3.join(transf4)
            transf4.join(transf3)
            transf1.join(word_analysis.SkipToTransformation("foo", "bar"))
            transf1.join(word_analysis.WordTransformationSequence([transf1]))

    def test_join_no_prefix(self) -> None:
        transf1 = word_analysis.EditTransformation("", "ugo", "il")
        transf2 = word_analysis.EditTransformation("", "ugo", "il")
        new_transf1 = transf1.join(transf2)
        new_transf2 = transf2.join(transf1)
        self.assertEquals(transf1, new_transf1)
        self.assertEquals(transf2, new_transf1)
        self.assertEquals(transf1, new_transf2)
        self.assertEquals(transf2, new_transf2)

    def test_join_prefix(self) -> None:
        transf1 = word_analysis.EditTransformation("abc", "ugo", "il")
        transf2 = word_analysis.EditTransformation("abc", "ugo", "il")
        transf3 = word_analysis.EditTransformation("edbc", "ugo", "il")
        self.assertEquals(transf1, transf1.join(transf1))
        self.assertEquals(transf1, transf1.join(transf2))
        self.assertEquals(transf1, transf2.join(transf1))
        expected23 = word_analysis.EditTransformation("bc", "ugo", "il")
        self.assertEquals(expected23, transf2.join(transf3))
        self.assertEquals(expected23, transf3.join(transf2))

    def test_apply_no_jump(self) -> None:
        transf = word_analysis.EditTransformation("ob", "a", "a")
        transformed, transformee = transf.apply_step("f", "obarfoobar")
        self.assertEqual(transformed, "foba")
        self.assertEqual(transformee, "rfoobar")

    def test_apply_with_jump(self) -> None:
        transf = word_analysis.EditTransformation("ob", "a", "a")
        transformed, transformee = transf.apply_step("f", "ooboobarfoobar")
        self.assertEqual(transformed, "foobooba")
        self.assertEqual(transformee, "rfoobar")


class SkipToTransformationTests(unittest.TestCase):

    def test_apply_unmatch(self) -> None:
        transf = word_analysis.SkipToTransformation("foo", "bar")
        self.assertRaises(ValueError, transf.apply_step, "pete", "fobare")
        self.assertRaises(ValueError, transf.apply_step, "pete", "foobae")
        self.assertRaises(ValueError, transf.apply_step, "pete", "fo0bae")

    def test_maybe_joinable(self) -> None:
        transf1 = word_analysis.SkipToTransformation("foo", "bar")
        transf2 = word_analysis.SkipToTransformation("abc", "def")
        self.assertTrue(transf1.maybe_joinable(transf1))
        self.assertTrue(transf2.maybe_joinable(transf2))
        self.assertTrue(transf1.maybe_joinable(transf2))
        self.assertTrue(transf2.maybe_joinable(transf1))
        self.assertFalse(transf1.maybe_joinable(word_analysis.EditTransformation("", "hugo", "herbert")))
        self.assertFalse(transf1.maybe_joinable(word_analysis.WordTransformationSequence([transf1])))

    def test_equals(self) -> None:
        transf1 = word_analysis.SkipToTransformation("foo", "bar")
        transf11 = word_analysis.SkipToTransformation("foo", "bar")
        transf2 = word_analysis.SkipToTransformation("abc", "def")
        self.assertEquals(transf1, transf1)
        self.assertEquals(transf1, transf11)
        self.assertNotEquals(transf1, transf2)
        self.assertNotEquals(transf1, word_analysis.EditTransformation("" , "hugo", "herbert"))
        self.assertNotEquals(transf1, word_analysis.WordTransformationSequence([transf1]))

    def test_join_(self) -> None:
        transf1 = word_analysis.SkipToTransformation("foo", "bar")
        transf2 = word_analysis.SkipToTransformation("o", "rab")
        new_transf1 = transf1.join(transf2)
        new_transf2 = transf2.join(transf1)
        expected_transf = word_analysis.SkipToTransformation("o", "")
        self.assertEquals(expected_transf, new_transf1)
        self.assertEquals(expected_transf, new_transf2)

    def test_join_unjoinable(self) -> None:
        transf = word_analysis.SkipToTransformation("foo", "bar")
        with self.assertRaises(ValueError):
            transf.join(word_analysis.EditTransformation("", "foo", "bar"))
            transf.join(word_analysis.WordTransformationSequence([transf]))

class TransformationSequenceTests(unittest.TestCase):

    def test_apply(self) -> None:
        transf = word_analysis.WordTransformationSequence(
            [word_analysis.EditTransformation("f", "uncti", ""),
             word_analysis.EditTransformation("o", "n", "o"),
             word_analysis.EditTransformation("", "", "bar")]
        )
        transformed, transformee = transf.apply_step("", "function")
        self.assertEqual(transformed, "foobar")
        self.assertEqual(transformee, "")

    def test_maybe_joinable(self) -> None:
        subt1 = word_analysis.EditTransformation("", "", "ge")
        subt1f = word_analysis.EditTransformation("", "", "hugo")
        subt2a = word_analysis.SkipToTransformation("foo", "bar")
        subt2b = word_analysis.SkipToTransformation("o", "bah")
        transf1 = word_analysis.WordTransformationSequence([subt1, subt2a])
        transf11 = word_analysis.WordTransformationSequence([subt1, subt2a])
        transf2 = word_analysis.WordTransformationSequence([subt1, subt2b])
        transf3 = word_analysis.WordTransformationSequence([subt1f, subt2a])
        transf4 = word_analysis.WordTransformationSequence([subt1])
        transf5 = word_analysis.WordTransformationSequence([subt2a, subt1])
        self.assertTrue(word_analysis.WordTransformationSequence([]).maybe_joinable(word_analysis.WordTransformationSequence([])))
        self.assertTrue(transf1.maybe_joinable(transf1))
        self.assertTrue(transf1.maybe_joinable(transf11))
        self.assertTrue(transf11.maybe_joinable(transf1))
        self.assertTrue(subt2a.maybe_joinable(subt2b))
        self.assertTrue(transf1.maybe_joinable(transf2))
        self.assertTrue(transf2.maybe_joinable(transf1))
        self.assertTrue(transf2.maybe_joinable(transf11))
        self.assertFalse(transf1.maybe_joinable(transf3))
        self.assertFalse(transf3.maybe_joinable(transf1))
        self.assertFalse(transf1.maybe_joinable(transf4))
        self.assertFalse(transf4.maybe_joinable(transf1))
        self.assertFalse(transf1.maybe_joinable(transf5))
        self.assertFalse(transf5.maybe_joinable(transf1))
        self.assertFalse(transf1.maybe_joinable(subt1))
        self.assertFalse(transf1.maybe_joinable(subt2a))

    def test_maybe_joinable_different_lengths(self) -> None:
        # todo: support for this needs to be implemented
        subt1 = word_analysis.EditTransformation("", "", "foo")
        subt2 = word_analysis.SkipToTransformation("asdf", "foo")
        transf1 = word_analysis.WordTransformationSequence([subt1])
        transf2 = word_analysis.WordTransformationSequence([subt2, subt1])
        self.assertTrue(transf1.maybe_joinable(transf2))
        self.assertTrue(transf2.maybe_joinable(transf1))

    def test_maybe_joinable_single_elements(self) -> None:
        # todo: support for this needs to be implemented
        # todo: needs tests for vice versa
        subt1 = word_analysis.EditTransformation("", "foo", "bar")
        subt2 = word_analysis.SkipToTransformation("bar", "foo")
        transf1 = word_analysis.WordTransformationSequence([subt1])
        transf2 = word_analysis.WordTransformationSequence([subt2])
        self.assertTrue(transf1.maybe_joinable(subt1))
        self.assertTrue(transf2.maybe_joinable(subt2))
        # allow SkipToTransformation objects to be a prefix or postfix in a sequence
        transf3 = word_analysis.WordTransformationSequence([subt2, subt1])
        self.assertTrue(transf3.maybe_joinable(subt1))
        transf4 = word_analysis.WordTransformationSequence([subt1, subt2])
        self.assertTrue(transf4.maybe_joinable(subt1))
        # do not allow EditTransformation objects to be a prefix or postfix in a sequence
        self.assertFalse(transf3.maybe_joinable(subt2))
        self.assertFalse(transf4.maybe_joinable(subt2))


    def test_equals(self) -> None:
        subt1 = word_analysis.EditTransformation("", "", "ge")
        subt1f = word_analysis.EditTransformation("", "", "hugo")
        subt2a = word_analysis.SkipToTransformation("foo", "bar")
        subt2b = word_analysis.SkipToTransformation("o", "bah")
        transf1 = word_analysis.WordTransformationSequence([subt1, subt2a])
        transf11 = word_analysis.WordTransformationSequence([subt1, subt2a])
        transf2 = word_analysis.WordTransformationSequence([subt1, subt2b])
        transf3 = word_analysis.WordTransformationSequence([subt1f, subt2a])
        transf4 = word_analysis.WordTransformationSequence([subt1])
        transf5 = word_analysis.WordTransformationSequence([subt2a, subt1])
        self.assertEquals(transf1, transf1)
        self.assertEquals(transf1, transf11)
        self.assertEquals(transf11, transf1)
        self.assertNotEquals(transf2, transf1)
        self.assertNotEquals(transf1, transf2)
        self.assertNotEquals(transf11, transf2)
        self.assertNotEquals(transf2, transf11)
        self.assertNotEquals(transf1, transf3)
        self.assertNotEquals(transf1, transf4)
        self.assertNotEquals(transf1, transf5)

    def test_join_unjoinable(self) -> None:
        subt1 = word_analysis.EditTransformation("", "", "ge")
        subt1f = word_analysis.EditTransformation("", "", "hugo")
        subt2a = word_analysis.SkipToTransformation("foo", "bar")
        transf1 = word_analysis.WordTransformationSequence([subt1, subt2a])
        transf3 = word_analysis.WordTransformationSequence([subt1f, subt2a])
        transf4 = word_analysis.WordTransformationSequence([subt1])
        transf5 = word_analysis.WordTransformationSequence([subt2a, subt1])
        with self.assertRaises(ValueError):
            transf1.join(transf3)
            transf3.join(transf1)
            transf1.join(transf4)
            transf4.join(transf1)
            transf1.join(transf5)
            transf5.join(transf1)

    def test_join(self) -> None:
        subt1 = word_analysis.EditTransformation("", "", "ge")
        subt2a = word_analysis.SkipToTransformation("foo", "bar")
        subt2b = word_analysis.SkipToTransformation("o", "bah")
        transf1 = word_analysis.WordTransformationSequence([subt1, subt2a])
        transf11 = word_analysis.WordTransformationSequence([subt1, subt2a])
        transf2 = word_analysis.WordTransformationSequence([subt1, subt2b])
        expected = word_analysis.WordTransformationSequence([
            word_analysis.EditTransformation("", "", "ge"),
            word_analysis.SkipToTransformation("o", "ba")
        ])
        joined1 = transf1.join(transf2)
        joined2 = transf2.join(transf1)
        self.assertEquals(expected, joined1)
        self.assertEquals(expected, joined2)
        joined11 = transf1.join(transf11)
        self.assertEquals(transf1, joined11)