"""Runs Linear Regression for a synthetic dataset."""

from pathlib import Path
from typing import Callable, Dict

import flwr as fl
from flwr.common.parameter import ndarrays_to_parameters
import hydra
import numpy as np
from omegaconf import DictConfig

from strategy import FedExP
import client, utils

def get_on_fit_config_fn(base_client_learning_rate: float) -> Callable[[int], Dict[str, float]]:
    def fit_config(server_round: int) -> Dict[str, float]:
        config = {
            "client_learning_rate": base_client_learning_rate
        }
        return config
    return fit_config

@hydra.main(config_path="docs/conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run CNN federated learning on FEMNIST.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    client_fn, testDataset, num_clients = client.gen_client_fn(
        num_clients=cfg.num_clients,
        model_size=cfg.model_size,
        num_samples_per_client=cfg.num_samples_per_client,
        alpha=cfg.alpha,
        beta = cfg.beta,
        num_epochs=cfg.num_epochs
    )
    client_fraction_fit = 1

    evaluate_fn = utils.gen_evaluate_fn(testDataset=testDataset)

    # Generate some initial model parameters, just all 0s
    initial_parameters = ndarrays_to_parameters([np.zeros(cfg.model_size)])

    strategy = FedExP(
        fraction_fit=client_fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=utils.weighted_average,
        on_fit_config_fn=get_on_fit_config_fn(cfg.client_learning_rate),
        ε = cfg.epsilon,
        numberOfRoundsToEvaluateOver=cfg.numberOfRoundsToEvaluateOver
    )

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    file_suffix: str = (
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_ηl={cfg.client_learning_rate}"
        f"_ε={cfg.epsilon}"
        f"_evaluateRounds={cfg.numberOfRoundsToEvaluateOver}"
        f"_run_num={cfg.run_num}"
    )

    np.save(
        Path(cfg.save_path) / Path(f"hist{file_suffix}"),
        history,  # type: ignore
    )

if __name__ == "__main__":
    main()
