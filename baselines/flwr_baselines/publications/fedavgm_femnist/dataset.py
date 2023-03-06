"""FEMNIST dataset utilities for federated learning."""


from typing import List, Optional, Tuple
from typing import *

from PIL import Image
from PIL.Image import Image as ImageType

import tarfile
import gdown
import csv
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from pathlib import Path

# Copied from Lab 1
class FEMNIST(Dataset):
    def __init__(
        self,
        mapping: Path,
        data_dir: Path,
        name: str = 'train',
        transform: Optional[Callable[[ImageType], Any]] = None,
        target_transform: Optional[Callable[[int], Any]] = None,
    ):
        """Function to initialise the FEMNIST dataset.

        Args:
            mapping (Path): path to the mapping folder containing the .csv files.
            data_dir (Path): path to the dataset folder. Defaults to data_dir.
            name (str): name of the dataset to load, train or test.
            transform (Optional[Callable[[ImageType], Any]], optional): transform function to be applied to the ImageType object. Defaults to None.
            target_transform (Optional[Callable[[int], Any]], optional): transform function to be applied to the label. Defaults to None.
        """
        self.data_dir = data_dir
        self.mapping = mapping
        self.name = name

        self.data: Sequence[Tuple[str, int]] = self._load_dataset()
        self.transform: Optional[Callable[[ImageType], Any]] = transform
        self.target_transform: Optional[Callable[[int], Any]] = target_transform

    def __getitem__(self, index) -> Tuple[Any, Any]:
        """Function used by PyTorch to get a sample.

        Args:
            index (_type_): index of the sample.

        Returns:
            Tuple[Any, Any]: couple (sample, label).
        """
        sample_path, label = self.data[index]

        # Convert to the full path
        full_sample_path: Path = self.data_dir / self.name / sample_path

        img: ImageType = Image.open(full_sample_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        """Function used by PyTorch to get the length of the dataset as the number of samples.

        Returns:
            int: the length of the dataset.
        """
        return len(self.data)
    
    def _load_dataset(self) -> Sequence[Tuple[str, int]]:
        """Load the paths and labels of the partition
        Preprocess the dataset for faster future loading
        If opened for the first time.

        Raises:
            ValueError: raised if the mapping file does not exist.

        Returns:
            Sequence[Tuple[str, int]]: partition asked as a sequence of couples (path_to_file, label)
        """
        preprocessed_path: Path = (self.mapping/self.name).with_suffix(".pt")
        if preprocessed_path.exists():
            return torch.load(preprocessed_path)
        else:
            csv_path = (self.mapping/self.name).with_suffix(".csv")
            if not csv_path.exists():
                raise ValueError(f"Required files do not exist, path: {csv_path}")
            else:
                with open(csv_path, mode="r") as csv_file:
                    csv_reader = csv.reader(csv_file)
                    # Ignore header
                    next(csv_reader)

                    # Extract the samples and the labels
                    partition: Sequence[Tuple[str, int]] = [
                        (sample_path, int(label_id))
                        for _, sample_path, _, label_id in csv_reader
                    ]

                    # Save for future loading
                    torch.save(partition, preprocessed_path)
                    return partition

def load_datasets(
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates the dataloaders to be fed into the model.

    Parameters
    ----------
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the DataLoader for testing.
    """
    random.seed(seed)
    home_dir = Path(__file__)
    home_dir = home_dir.parent
    dataset_dir: Path = home_dir / "dataset"
    data_dir: Path = dataset_dir / "data"
    centralized_mapping: Path = dataset_dir / 'client_data_mappings' / 'centralized' / '0'
    federated_partition: Path = dataset_dir / 'client_data_mappings' / 'fed_natural'
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    _download_data(home_dir, dataset_dir)
    
    train_datasets = []
    val_datasets = []
    for current_id in range(3230):
        full_file = federated_partition / str(current_id)
        dataset = FEMNIST(full_file, data_dir, 'train', transform)
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        ds_train = dataset # Remove this line when you actually want to have validation sets
        if len(ds_train) > batch_size:
            train_datasets.append(ds_train)
            val_datasets.append(ds_val)

    # Get the centralised test set
    testset = FEMNIST(centralized_mapping, data_dir, 'test', transform)

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    # Need to loop through all the potential clients
    for i in range(len(train_datasets)):
        trainloaders.append(DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(val_datasets[i], batch_size=batch_size))
    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)


def _download_data(home_dir, dataset_dir):
    """Downloads (if necessary) the FEMNIST dataset.
    """
    #  Download compressed dataset
    if not (home_dir / "femnist.tar.gz").exists():
        id = "1-CI6-QoEmGiInV23-n_l6Yd8QGWerw8-"
        gdown.download(
            f"https://drive.google.com/uc?export=download&confirm=pbef&id={id}",
            str(home_dir / "femnist.tar.gz"),
        )
        
    # Decompress dataset 
    if not dataset_dir.exists():
        file = tarfile.open("femnist.tar.gz")
        file.extractall(dataset_dir)
        print(f"Dataset extracted in {dataset_dir}")