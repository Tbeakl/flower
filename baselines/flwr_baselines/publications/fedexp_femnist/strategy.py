"""FEDEXP: SPEEDING UP FEDERATED AVERAGING VIA
EXTRAPOLATION [Jhunjhunwala et al., 2019] strategy.

Paper: https://arxiv.org/pdf/2106.04502.pdf
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate
from flwr.server.strategy.fedavg import FedAvg

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

# flake8: noqa: E501
class FedExP(FedAvg):
    """Configurable FedAvg with Momentum strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        ε: float = 0.1,
    ) -> None:
        """FEDEXP: SPEEDING UP FEDERATED AVERAGING VIA EXTRAPOLATION strategy

        Implementation based on https://arxiv.org/pdf/2106.04502.pdf

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.ε = ε

    def __repr__(self) -> str:
        rep = f"FedExp(accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def compute_norm_squared(self, update: NDArrays) -> float:
        """Compute the l2 norm of a parameter update with mismatched np array shapes, to be used in clipping"""
        flat_update = update[0]
        for i in range(1, len(update)):
            flat_update = np.append(flat_update, update[i])  # type: ignore
        squared_update = np.square(flat_update)
        norm_sum = np.sum(squared_update)
        return norm_sum

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using unweighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        update_results = [
            (parameters_to_ndarrays(fit_res.parameters), 1) for _, fit_res in results
        ]

        aggregated_update = aggregate(update_results)
        squared_norm_aggregated_update = self.compute_norm_squared(aggregated_update)
        assert (
            self.initial_parameters is not None
        ), "When using server-side optimization, model needs to be initialized."
        initial_weights = parameters_to_ndarrays(self.initial_parameters)


        global_learning_rate = 0
        M = len(update_results)
        for (update, _) in update_results:
            squared_norm_update = self.compute_norm_squared(update)
            global_learning_rate += squared_norm_update
        global_learning_rate /= (2 * M * (squared_norm_aggregated_update + self.ε))
        print(f"Alternative value suggested {global_learning_rate}")
        global_learning_rate = max(1, global_learning_rate)
        print(f"Global Value used: {global_learning_rate}")

        # SGD
        fedexp_result = [
            x - global_learning_rate * y
            for x, y in zip(initial_weights, aggregated_update)
        ]
        # Update current weights
        self.initial_parameters = ndarrays_to_parameters(fedexp_result)

        parameters_aggregated = ndarrays_to_parameters(fedexp_result)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Also add the server learning rate to the metrics which are being recorded

        return parameters_aggregated, metrics_aggregated
