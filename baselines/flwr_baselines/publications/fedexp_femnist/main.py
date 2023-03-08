"""Runs CNN federated learning for MNIST dataset."""

from pathlib import Path

import flwr as fl
from flwr.common.parameter import ndarrays_to_parameters
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from strategy import FedExP
import client, utils, model

DEVICE: torch.device = torch.device("cpu")

@hydra.main(config_path="docs/conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run CNN federated learning on FEMNIST.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """


    client_fn, testloader, num_clients = client.gen_client_fn(
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        device=DEVICE,
        learning_rate=cfg.client_learning_rate,
        learning_rate_decay=cfg.client_learning_rate_decay,
        gradient_clipping=cfg.gradient_clipping,
        max_norm=cfg.max_norm
    )

    client_fraction_fit = cfg.num_participating_clients / num_clients

    evaluate_fn = utils.gen_evaluate_fn(testloader, DEVICE)

    initialNet = model.Net()
    seed_model_params = [val.cpu().numpy() for _, val in initialNet.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(seed_model_params)

    strategy = FedExP(
        fraction_fit=client_fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=int(cfg.num_participating_clients),
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=utils.weighted_average,
        ε = cfg.epsilon
    )


    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    file_suffix: str = (
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_ηl={cfg.client_learning_rate}"
        f"_ε={cfg.epsilon}"
    )

    np.save(
        Path(cfg.save_path) / Path(f"hist{file_suffix}"),
        history,  # type: ignore
    )

    utils.plot_metric_from_history(
        history,
        cfg.save_path,
        cfg.expected_maximum,
        file_suffix,
    )


if __name__ == "__main__":
    main()
