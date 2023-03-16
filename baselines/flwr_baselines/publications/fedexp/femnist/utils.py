"""Util functions for FEMNIST
"""
import csv
import tarfile
from collections import OrderedDict
from pathlib import Path
from typing import *
from typing import Callable, Dict, Optional, Tuple

import gdown
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from PIL.Image import Image as ImageType
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import NDArrays, Parameters, Scalar
from flwr.server.history import History


class Net(nn.Module):
    """Convolutional Neural Network architecture as described in Reddi 2021
    paper :

    [ADAPTIVE FEDERATED OPTIMIZATION] (https://arxiv.org/pdf/2003.00295.pdf)
    """

    def __init__(self) -> None:

        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dense1 = nn.Linear(9216,128)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, 62)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        output_tensor = F.relu(self.conv1(input_tensor))
        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = self.dropout1(output_tensor)
        output_tensor = torch.flatten(output_tensor, start_dim=1)  # flatten all dimensions except batch
        output_tensor = F.relu(self.dense1(output_tensor))
        output_tensor = self.dropout2(output_tensor)
        output_tensor = self.dense2(output_tensor)
        return output_tensor

#This is taken from Lab 1 as a method for dealing with the Federated EMNIST dataset
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

def get_femnist_transform() -> Callable[[ImageType], Any]:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

def get_femnist_model() -> Module:
    """Generates a CNN for the CR task

    Args:

    Returns:
        Module: CNN.
    """
    return Net()

def gen_femnist_partitions(
    path_original_dataset: Path,
    dataset_name: str,
) -> Path:
    """Defines root path for partitions and calls functions to create them, and download the data

    Args:
        path_original_dataset (Path): Path to original (unpartitioned) dataset.
        dataset_name (str): Friendly name to dataset.

    Returns:
        Path: [description]
    """
    fed_dir = (
        path_original_dataset
        / f"{dataset_name}"
    )

    _download_data(path_original_dataset)

    return fed_dir

def _download_data(path_original_dataset):
    """Downloads (if necessary) the FEMNIST dataset.
    """
    if not (path_original_dataset / "femnist").exists():
        #  Download compressed dataset
        if not (path_original_dataset / "femnist.tar.gz").exists():
            id = "1-CI6-QoEmGiInV23-n_l6Yd8QGWerw8-"
            gdown.download(
                f"https://drive.google.com/uc?export=download&confirm=pbef&id={id}",
                str(path_original_dataset / "femnist.tar.gz"),
            )

        # Decompress dataset  
        with tarfile.open("femnist.tar.gz") as file:
            file.extractall(path_original_dataset)
        print(f"Dataset extracted in {path_original_dataset}")

def train(
    net: Module,
    trainloader: DataLoader,
    epochs: int,
    device: str,
    learning_rate: float = 0.01,
    weight_decay: float = 0,
    gradient_clipping: bool = False,
    max_norm: float = 10,
) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=max_norm)
            optimizer.step()


def test(net: Module, testloader: DataLoader, device: str) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def gen_on_fit_config_fn(
    epochs_per_round: int,
    batch_size: int,
    client_learning_rate: float,
    client_learning_rate_decay: float,
    weight_decay: float,
    gradient_clipping: bool,
    max_norm: float
) -> Callable[[int], Dict[str, Scalar]]:
    """Generates ` On_fit_config`

    Args:
        epochs_per_round (int):  number of local epochs.
        batch_size (int): Batch size
        client_learning_rate (float): Learning rate of client
        client_learning_rate_decay (float): Value multiped onto the client learning rate each round
        weight_decay (float): The value for weight decay in the local optimizer
        gradient_clipping (bool): Should the local optimizer use gradient clipping
        max_norm (float): If gradient clipping is used then this specifies the max norm passed in
    Returns:
        Callable[[int], Dict[str, Scalar]]: Function to be called at the beginning of each rounds.
    """

    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with specific client learning rate."""
        local_config: Dict[str, Scalar] = {
            "epoch_global": server_round,
            "epochs_per_round": epochs_per_round,
            "batch_size": batch_size,
            "client_learning_rate": client_learning_rate * (client_learning_rate_decay ** server_round),
            "weight_decay": weight_decay,
            "gradient_clipping": gradient_clipping,
            "max_norm": max_norm
        }
        return local_config

    return on_fit_config


def get_femnist_eval_fn(
    path_original_dataset: Path
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Returns an evaluation function for centralized evaluation."""

    testset = FEMNIST(path_original_dataset / "femnist" / 'client_data_mappings' / 'centralized' / '0', path_original_dataset / "femnist" / "data", 'test', transform=get_femnist_transform())

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire FEMNIST test set for evaluation."""
        # determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = get_femnist_model()
        state_dict = OrderedDict(
            {
                k: torch.tensor(np.atleast_1d(v))
                for k, v in zip(net.state_dict().keys(), parameters_ndarrays)
            }
        )
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


def get_initial_parameters() -> Parameters:
    """Returns initial parameters from a model.

    Args:
        num_classes (int, optional): Defines if using CIFAR10 or 100. Defaults to 10.

    Returns:
        Parameters: Parameters to be sent back to the server.
    """
    model = get_femnist_model()
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(weights)

    return parameters


def plot_metric_from_history(
    hist: History,
    dataset_name: str,
    strategy_name: str,
    expected_maximum: float,
    save_plot_path: Path,
) -> None:
    """Simple plotting method for Classification Task.

    Args:
        hist (History): Object containing evaluation for all rounds.
        dataset_name (str): Name of the dataset.
        strategy_name (str): Strategy being used
        expected_maximum (float): Expected final accuracy.
        save_plot_path (Path): Where to save the plot.
    """
    rounds, values = zip(*hist.metrics_centralized["accuracy"])
    plt.figure()
    plt.plot(rounds, np.asarray(values) * 100, label=strategy_name)  # Accuracy 0-100%
    # Set expected graph
    plt.axhline(y=expected_maximum, color="r", linestyle="--")
    plt.title(f"Centralized Validation - {dataset_name}")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(save_plot_path)
    plt.close()
