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