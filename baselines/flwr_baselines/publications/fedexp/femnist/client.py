"""Flower Client for FEMNIST."""
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common.typing import NDArrays, Scalar

from .utils import FEMNIST, get_femnist_model, get_femnist_transform, test, train


class RayClient(fl.client.NumPyClient):
    """Ray Virtual Client."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainset: FEMNIST,
        valset: FEMNIST,
    ):
        self.net = net
        self.trainset = trainset
        self.valset = valset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        trainloader = DataLoader(self.trainset, batch_size=int(config["batch_size"]))

        train(
            net=self.net,
            trainloader=trainloader,
            epochs=int(config["epochs_per_round"]),
            device=self.device,
            learning_rate=config['client_learning_rate'],
            weight_decay=config['weight_decay'],
            gradient_clipping=config['gradient_clipping'],
            max_norm=config['max_norm']
        )

        return self.get_parameters({}), len(trainloader), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        valloader = DataLoader(self.valset, batch_size=int(config["batch_size"]))
        loss, accuracy = test(self.net, valloader, self.device)
        return float(loss), len(valloader), {"accuracy": float(accuracy)}

def get_ray_client_fn(
    fed_dir: Path,
) -> Callable[[str], RayClient]:
    """Function that loads a Ray (Virtual) Client.

    Args:
        fed_dir (Path): Path containing local datasets in the form ./client_id/train.pt

    Returns:
        Callable[[str], RayClient]: [description]
    """

    net = get_femnist_model()

    def client_fn(cid: str) -> RayClient:
        # create a single client instance
        trainset = FEMNIST(fed_dir / 'client_data_mappings' / 'fed_natural' / cid, fed_dir / 'data', 'train', transform=get_femnist_transform())
        testset = FEMNIST(fed_dir / 'client_data_mappings' / 'fed_natural' / cid, fed_dir / 'data', 'test', transform=get_femnist_transform())
        return RayClient(
            net,
            trainset,
            testset,
        )

    return client_fn