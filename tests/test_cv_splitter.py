import unittest

from biotrainer.trainers.cv_splitter import CrossValidationSplitter


class CrossValidationSplitterTests(unittest.TestCase):
    def test_create_bins_from_continuous_values(self):
        bins1 = CrossValidationSplitter._continuous_values_to_bins([1])
        self.assertTrue(len(bins1) == 1, "Bins creation for single value failed!")

        bins2 = CrossValidationSplitter._continuous_values_to_bins([x for x in range(0, 10000, 1)])
        bins3 = CrossValidationSplitter._continuous_values_to_bins([x for x in range(0, 20000, 2)])
        self.assertTrue(len(set(bins2)) == len(set(bins3)), "Bins set size is not equal, "
                                                            "but it must be equal for all inputs!")

        bins4 = CrossValidationSplitter._continuous_values_to_bins([x for x in range(-10000, 50000, 3)])
        self.assertTrue(len(bins4) == len(list(range(-10000, 50000, 3))), "Bins creation with negative numbers failed!")


if __name__ == '__main__':
    unittest.main()
