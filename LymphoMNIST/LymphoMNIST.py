import os
import sys
import requests
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import _flip_byte_order, check_integrity, extract_archive
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import torch
import numpy as np
import codecs
from tqdm import tqdm 

# Utility function to read and convert byte data to integers
def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


# Mapping of SN3_PASCALVINCENT_TYPEMAP for tensor dtype conversion
SN3_PASCALVINCENT_TYPEMAP = {
    8: torch.uint8,
    9: torch.int8,
    11: torch.int16,
    12: torch.int32,
    13: torch.float32,
    14: torch.float64,
}

# Function to read tensor data from a file in Pascal Vincent's format
def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    with open(path, "rb") as f:
        data = f.read()
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]

    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))
    if sys.byteorder == "little" and parsed.element_size() > 1:
        parsed = _flip_byte_order(parsed)
    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)

# Functions to read label and image files, refactored for clarity and error handling
def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8 or x.ndimension() != 1:
        raise ValueError("Invalid format for label file.")
    return x.long()

def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8 or x.ndimension() != 3:
        raise ValueError("Invalid format for image file.")
    return x

# Custom VisionDataset class for LymphoMNIST dataset
class LymphoMNIST(VisionDataset):
    # Updated comments for clarity and added more descriptive class documentation
    mirrors = [
        "https://www.dropbox.com/scl/fo/nsk5xah661jxhp7meftov/h?rlkey=bt1wk4b9djfl2uzj8brf7le43&dl=0",
        # "other mirrors...",
    ]
    resources = [("train_images", None), ("train_labels", None), ("test_images", None), ("test_labels", None)]

    def __init__(self, 
                 root: str, 
                 train: bool = True, 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None, 
                 download: bool = False, 
                 num_classes: int = 3) -> None:
        
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.num_classes = num_classes
        self.classes = ["B", "T"] if num_classes == 2 else ["B", "T4", "T8"]
        if download:
            self.download()
            
        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it.")
        self.data, self.targets = self._load_data()
    
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = f"{'train' if self.train else 'test'}_images"
        label_file = f"{'train' if self.train else 'test'}_labels"
        data_path = os.path.join(self.raw_folder, image_file)
        label_path = os.path.join(self.raw_folder, label_file)
        data = read_image_file(data_path)
        targets = read_label_file(label_path)
        if self.num_classes == 2:
            # Combine labels for "T4" and "T8" into a single class
            targets = targets.apply_(lambda x: x if x == 0 else 1)
        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Simplified image loading and conversion for consistency
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self) -> bool:
        # Simplified file existence check
        return all(check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0])) for url, _ in self.resources)


    def download(self):
        if self._check_exists():
            print("Dataset already exists. Skipping download.")
            return
        os.makedirs(self.raw_folder, exist_ok=True)
        for url in self.mirrors:
            try:
                direct_link = url.replace('dl=0', 'dl=1').replace('view?usp=sharing', 'uc?export=download')
                response = requests.get(direct_link, stream=True)
                if response.status_code == 200:
                    # Determine the total size of the file to download
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 1024  # 1 Kibibyte
                    
                    # Open the file to write as binary
                    download_path = os.path.join(self.raw_folder, "idx_files.zip")
                    print(f"Downloading {url} to  {download_path}")
                    with open(download_path, 'wb') as f, tqdm(
                            total=total_size, unit='iB', unit_scale=True,
                            desc=download_path) as bar:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                                # Update the progress bar
                                bar.update(len(chunk))
                    extract_archive(download_path, self.raw_folder)
                    print("Dataset downloaded and extracted successfully.")
                    break
            except Exception as e:
                print(f"An error occurred: {e}. Trying next mirror...")
        else:
            raise Exception("All mirrors failed. Please check your internet connection or the mirror URLs.")


    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")
