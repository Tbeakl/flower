"""FEMNIST dataset utilities for federated learning."""


from typing import List, Optional, Tuple
from typing import *
import numpy.typing as npt

import numpy as np
import random

def load_datasets(
    num_clients: int = 20,
    model_size: int = 1000,
    num_samples_per_client: int = 30,
    alpha: float = 0.1,
    beta: float = 0.1,
    seed: Optional[int] = 0,
) -> Tuple[List[Tuple[npt.NDArray,npt.NDArray]],  Tuple[npt.NDArray, npt.NDArray]]:
    """Creates the datasets to be fed into the model.

    Parameters
    ----------
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Tuple[NDArray,NDArray]],  Tuple[NDArray, NDArray]]
        The datasets for training by client and the overall dataset which can be used for testing
    """
    random.seed(seed)
    np.random.seed(seed)

    X_split, y_split = get_client_data(num_clients=num_clients, model_size=model_size, num_samples_per_user=num_samples_per_client, alpha=alpha, beta=beta)

    # We want to normalise all of these values
    X_split = [(X_i.T / np.linalg.norm(X_i, axis=1)).T for X_i in X_split]
    y_split = [y_i/np.linalg.norm(y_i) for y_i in y_split]

    client_datasets = [(X_split[i], y_split[i]) for i in range(num_clients)]
    test_dataset = (np.concatenate(X_split), np.concatenate(y_split))
    return client_datasets, test_dataset


# Copied with heavy tidying up and making easier to understand from https://github.com/Divyansh03/FedExP/blob/main/linear_regression.ipynb
def get_client_data(num_clients, model_size, num_samples_per_user, alpha, beta):
    number_of_classes = 1
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
        xx = np.random.multivariate_normal(mean_x[i], cov_x, num_samples_per_user)
        yy = np.dot(xx,W)
        X_split[i] = xx
        y_split[i] = yy.flatten()

    return X_split, y_split