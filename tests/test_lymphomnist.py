import unittest
from LymphoMNIST.LymphoMNIST import LymphoMNIST
from PIL import Image
import torch, torchvision, os

class TestLymphoMNIST(unittest.TestCase):
    def train_initialization(self):
        """Test that the dataset initializes without errors."""
        try:
            dataset = LymphoMNIST(root='./data', train=True, download=True)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Initialization failed: {e}")
    
    def train_length(self):
        """Test that the dataset has the expected number of items."""
        dataset = LymphoMNIST(root='./data', train=True, download=True)
        self.assertEqual(len(dataset), 64000)


    def train_data_types(self):
        dataset = LymphoMNIST(root='./data', train=True, download=True)
        img, label = dataset[0]
        self.assertIsInstance(img, Image.Image, "The image is not an instance of PIL.Image.Image")
        self.assertIsInstance(label, int, "The label is not of type int")
    
    
    def train_data_range(self):
        dataset = LymphoMNIST(root='./data', train=True, download=True)
        img, label = dataset[0]
        # Assuming labels are integers representing classes
        self.assertIn(label, range(dataset.num_classes), "Label is out of expected range")
        
        
    def train_transformations(self):
        transform = torchvision.transforms.ToTensor()
        dataset = LymphoMNIST(root='./data', train=True, download=True, transform=transform)
        img, _ = dataset[0]
        self.assertIsInstance(img, torch.Tensor, "Transform not applied: Image is not a torch.Tensor")
        
    def test_download(self):
        # Ensure the data directory is clean before this test or handle cleanup
        dataset = LymphoMNIST(root='./temp_data', train=False, download=True)
        self.assertTrue(os.path.exists('./temp_data/LymphoMNIST/raw'), "Data not downloaded correctly")


    def test_dataset_length(self):
        dataset = LymphoMNIST(root='./data', train=False, download=True)
        expected_length = 16000  # Replace with the actual expected length
        self.assertEqual(len(dataset), expected_length, "Dataset reported length does not match expected")

    
    def test_invalid_path(self):
        with self.assertRaises(RuntimeError):
            LymphoMNIST(root='./non_existent_path', train=False, download=False)
    


if __name__ == '__main__':
    unittest.main()
