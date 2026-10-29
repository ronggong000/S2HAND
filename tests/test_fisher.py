import unittest

from tools.probe_fisher import parse_blocks


class FisherUtilityTest(unittest.TestCase):
    def test_all_blocks(self):
        self.assertEqual(parse_blocks("all", 4), [0, 1, 2, 3])

    def test_multiple_separators(self):
        self.assertEqual(parse_blocks("4:38:40", 56), [4, 38, 40])
        self.assertEqual(parse_blocks("1,2;3", 4), [1, 2, 3])

    def test_rejects_duplicate_or_out_of_range(self):
        with self.assertRaises(ValueError):
            parse_blocks("2:2", 4)
        with self.assertRaises(ValueError):
            parse_blocks("4", 4)


if __name__ == "__main__":
    unittest.main()
