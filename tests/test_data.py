"""
Unit tests for data loading.
"""

import unittest
from data.snare_dataset import SNAREDataset


class TestData(unittest.TestCase):

    def test_dataset_length(self):
        dataset = SNAREDataset("data/snare.csv")
        self.assertTrue(len(dataset) > 0)

    def test_item_structure(self):
        dataset = SNAREDataset("data/snare.csv")
        item = dataset.get_item(0)
        self.assertIn("features", item)
        self.assertIn("label", item)


if __name__ == "__main__":
    unittest.main()