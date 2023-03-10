# pylint: disable=too-many-arguments
"""Defines the MNIST Flower Client and a function to instantiate it."""


from collections import OrderedDict
from typing import Callable, Dict, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

import model
from dataset import load_datasets
from copy import deepcopy

class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        gradient_clipping: bool,
        max_norm: float
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_clipping = gradient_clipping
        self.max_norm = max_norm

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)
        model.train(
            self.net,
            self.trainloader,
            self.device,
            epochs=self.num_epochs,
            learning_rate=config['learning_rate'],
            gradient_clipping=self.gradient_clipping,
            max_norm=self.max_norm
        )

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = model.test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    gradient_clipping: bool,
    max_norm: float
) -> Tuple[Callable[[str], FlowerClient], DataLoader, int]:
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    device : torch.device
        The device on which the the client will train on and test on.
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    batch_size : int
        The size of the local batches each client trains on.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader, int]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing and the number of clients available
    """
    trainloaders, valloaders, testloader = load_datasets(
        batch_size=batch_size
    )

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        net = model.Net().to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(
            net, trainloader, valloader, device, num_epochs, gradient_clipping=gradient_clipping, max_norm=max_norm
        )

    return client_fn, testloader, len(trainloaders)