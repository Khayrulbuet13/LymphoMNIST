import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataset import ConcatDataset, Dataset
from torch.utils.data import random_split
from typing import Tuple, List

def split_dataset(train_dataset: Dataset, 
                  test_dataset: Dataset, 
                  split_ratio: List[float], 
                  random_seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Combines training and testing datasets and then splits the combined dataset into
    training, validation, and testing sets based on the defined split ratios.
    
    Parameters:
    - train_dataset: The original training dataset.
    - test_dataset: The original testing dataset.
    - split_ratio: A list of three values indicating the proportion of the dataset
      to allocate to the training, validation, and testing sets, respectively. The values should sum up to 1.
    - random_seed: Seed for reproducibility of the split.
    
    Returns:
    - train_set: The subset for training.
    - val_set: The subset for validation.
    - test_set: The subset for testing.
    """
    
    # Ensure the split ratios sum up to 1
    assert sum(split_ratio) == 1, "The split ratios must sum up to 1."
    
    # Combine the training and testing datasets
    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    
    # Calculate the sizes for each split
    total_size = len(combined_dataset)
    train_size = int(total_size * split_ratio[0])
    val_size = int(total_size * split_ratio[1])
    test_size = total_size - train_size - val_size  # Ensure no data is lost due to rounding
    
    # Perform the split
    train_set, val_set, test_set = random_split(combined_dataset, [train_size, val_size, test_size], 
                                                generator=torch.Generator().manual_seed(random_seed))
    
    return train_set, val_set, test_set




def plot_dl(dl, labels_map=None, n=3):
    figure = plt.figure(figsize=(10, 10))  # Set figure size for better visibility
    cols, rows = n, n
    
    # Ensure we get n*n images; if the dataloader doesn't have enough, reduce n accordingly
    total_images = min(len(dl.dataset), n*n)
    cols, rows = n, min(total_images, n)
    
    for i in range(1, cols * rows + 1):
        # Directly accessing the dataset for random samples
        sample_idx = torch.randint(len(dl.dataset), size=(1,)).item()
        img, label = dl.dataset[sample_idx]
        
        figure.add_subplot(rows, cols, i)
        if labels_map:
            plt.title(labels_map[label.item()])
        else:
            plt.title(label.item())
        plt.axis("off")
        if img.shape[0] == 3:  # If it's a color image
            plt.imshow(img.permute(1, 2, 0))  # Convert from CxHxW to HxWxC
        else:  # If it's a grayscale image
            plt.imshow(img.squeeze(), cmap="gray")  # Remove channel dim if it's present
    
    plt.show()





