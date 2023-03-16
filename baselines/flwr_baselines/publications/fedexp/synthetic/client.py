# pylint: disable=too-many-arguments
"""Defines the Synthetic Flower Client and a function to instantiate it."""

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import numpy.typing as npt

import flwr as fl
from flwr.common.typing import NDArrays, Scalar


class FlowerClient(fl.client.NumPyClient):
    """Flower client for Linear Regression training."""

    def __init__(
        self,
        A: npt.NDArray,
        b: npt.NDArray,
    ):
        self.A = A
        self.b = b
        self.J = A.T.dot(A)
        self.e = A.T.dot(b)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the weights of the current model"""
        return [self.weights]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the weights of the model using the given ones."""
        self.weights = parameters[0]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)
        # print(f"Parameters: {parameters[0]}")
        # print(f"Epochs: {config['epochs_per_round']}")
        for _ in range(config['epochs_per_round']):
            grad = self.J.dot(self.weights) - self.e
            self.weights = self.weights - config['client_learning_rate'] * grad

        return self.get_parameters({}), len(self.b), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss = np.linalg.norm(np.matmul(self.A, self.weights) - self.b) ** 2
        return loss, len(self.b), {}


def get_ray_client_fn(
    fed_dir: Path
) -> Callable[[str], FlowerClient]:
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
        fed_dir: Path
            The directory containing all the partitions of data for the clients
    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader, int]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing and the number of clients available
    """
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        A = np.load(fed_dir / cid / "X.npy")
        b = np.load(fed_dir / cid / "y.npy")

        return FlowerClient(
            A=A, b=b
        )

    return client_fn