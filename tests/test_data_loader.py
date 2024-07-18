# tests/test_data_loader.py
import unittest
from src.data_loader import ProteinDataset

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.dataset = ProteinDataset('data/train.csv')

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), 1000)  # Replace 1000 with the actual number of samples

    def test_one_hot_encoding(self):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        encoded = self.dataset.one_hot_encode(sequence)
        self.assertEqual(encoded.shape, (len(sequence), 21))
        self.assertEqual(encoded.sum(), len(sequence))

if __name__ == '__main__':
    unittest.main()
