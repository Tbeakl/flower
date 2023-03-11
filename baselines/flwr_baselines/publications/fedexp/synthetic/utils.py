"""Util functions for CIFAR10/100."""
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import NDArrays, Parameters, Scalar
from flwr.server.history import History

def gen_federated_partitions(path_original_dataset: Path,
                            dataset_name: str,
                            num_total_clients: int,
                            num_samples_per_client: int,
                            alpha: float,
                            beta: float,
                            model_size: int,
                            seed: int = 42):
    """Defines root path for partitions and calls functions to create them.

    Args:
        path_original_dataset (Path): Path to original (unpartitioned) dataset.
        dataset_name (str): Friendly name to dataset.
        num_total_clients (int): Number of clients.
        num_samples_per_client: (int): Number of samples at each client
        alpha (float): Used in generation of data
        beta (float): Used in generation of data
        model_size (int): Number of parameters used by the model
        seed (int), Optional: Seed to be used in the random number generation of the dataset
    Returns:
        Path: [description]
    """

    fed_dir = (
        path_original_dataset
        / f"{dataset_name}"
        / "partitions"
        / f"{num_total_clients}"
        / f"{num_samples_per_client}"
        / f"{model_size}"
        / f"{alpha:.2f}"
        / f"{beta:.2f}"
    )

    (X, y) = get_synthetic_dataset( path_original_dataset=path_original_dataset,
                                    num_clients=num_total_clients,
                                    model_size=model_size,
                                    num_samples_per_client=num_samples_per_client,
                                    alpha=alpha,
                                    beta=beta,
                                    seed=seed)

    # Break up into the different clients, just split up evenly along the different client tracts and store down the data
    for i in range(num_total_clients):
        part_dir = fed_dir / f"{i}"
        part_dir.mkdir(exist_ok=True, parents=True)
        np.save(part_dir / "X", X[i*num_samples_per_client: (i + 1) * num_samples_per_client])
        np.save(part_dir / "y", y[i*num_samples_per_client: (i + 1) * num_samples_per_client])

    return fed_dir

def get_synthetic_dataset(  path_original_dataset: Path,
                            num_clients: int,
                            model_size: int,
                            num_samples_per_client: int,
                            alpha: float,
                            beta: float,
                            seed: int) -> Tuple[npt.NDArray, npt.NDArray]:
    np.random.seed(seed)
    full_dataset_path = (path_original_dataset /
                        "synthetic" 
                        / f"{num_clients}"
                        / f"{num_samples_per_client}"
                        / f"{model_size}"
                        / f"{alpha:.2f}"
                        / f"{beta:.2f}")
    
    if full_dataset_path.exists():
        # This means we can just load in the existing data
        return (np.load(full_dataset_path / "X.npy"), np.load(full_dataset_path / "y.npy"))
    
    full_dataset_path.mkdir(exist_ok=True, parents=True)
    number_of_classes = 1
    # Need to generate the actual data
    # This part has been copied, with heavy reformatting from https://github.com/Divyansh03/FedExP/blob/main/linear_regression.ipynb
    X_split = [[] for _ in range(num_clients)]
    y_split = [[] for _ in range(num_clients)]
    #### define some prior ####
    mean_W = np.random.normal(0, alpha, num_clients)
    B = np.random.normal(0, beta, num_clients)
    mean_x = np.zeros((num_clients, model_size))
    cov_x = np.eye(model_size)

    for i in range(num_clients):
        mean_x[i] = np.random.normal(B[i], 1, model_size)

    for i in range(num_clients):
        W = np.random.normal(mean_W[i], 1, (model_size, number_of_classes))
        xx = np.random.multivariate_normal(mean_x[i], cov_x, num_samples_per_client)
        yy = np.dot(xx,W)
        X_split[i] = xx
        y_split[i] = yy.flatten()

    X_split = [(X_i.T / np.linalg.norm(X_i, axis=1)).T for X_i in X_split]
    y_split = [y_i/np.linalg.norm(y_i) for y_i in y_split]

    X = np.concatenate(X_split)
    y = np.concatenate(y_split)
    np.save(full_dataset_path / "X", X)
    np.save(full_dataset_path / "y", y)
    return (X, y)



def gen_on_fit_config_fn(
    epochs_per_round: int, client_learning_rate: float, client_learning_rate_decay: float
) -> Callable[[int], Dict[str, Scalar]]:
    """Generates ` On_fit_config`

    Args:
        epochs_per_round (int):  number of local epochs.
        batch_size (int): Batch size
        client_learning_rate (float): Learning rate of clinet

    Returns:
        Callable[[int], Dict[str, Scalar]]: Function to be called at the beginnig of each rounds.
    """

    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with specific client learning rate."""
        local_config: Dict[str, Scalar] = {
            "epoch_global": server_round,
            "epochs_per_round": epochs_per_round,
            "client_learning_rate": client_learning_rate * (client_learning_rate_decay ** server_round),
        }
        return local_config

    return on_fit_config


def get_eval_fn(
    path_original_dataset: Path,
    num_total_clients: int,
    model_size: int,
    num_samples_per_client: int,
    alpha: float,
    beta: float
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generates the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    full_dataset_path = (path_original_dataset /
                        "synthetic" 
                        / f"{num_total_clients}"
                        / f"{num_samples_per_client}"
                        / f"{model_size}"
                        / f"{alpha:.2f}"
                        / f"{beta:.2f}")
    
    X = np.load(full_dataset_path / "X.npy")
    y = np.load(full_dataset_path / "y.npy")

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        loss = np.linalg.norm(np.matmul(X, parameters_ndarrays[0]) - y) ** 2
        loss /= len(y)
        # return statistics
        return loss, {}
    return evaluate

def get_initial_parameters(model_size: int) -> Parameters:
    return ndarrays_to_parameters([np.zeros(model_size)])

def plot_loss_from_history(
    hist: History,
    dataset_name: str,
    strategy_name: str,
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
    rounds, values = zip(*hist.losses_centralized)
    plt.figure()
    plt.plot(rounds, np.asarray(values), label=strategy_name)
    # Set expected graph
    plt.title(f"Centralized Validation - {dataset_name}")
    plt.xlabel("Rounds")
    plt.ylabel("Mean Squared Error")
    plt.yscale("log")
    plt.legend(loc="upper left")
    plt.savefig(save_plot_path)
    plt.close()
