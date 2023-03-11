"""FEDEXP: SPEEDING UP FEDERATED AVERAGING VIA
EXTRAPOLATION [Jhunjhunwala et al., 2019] strategy.

Paper: https://arxiv.org/pdf/2106.04502.pdf
"""


from copy import deepcopy
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


from flwr.common import (
    EvaluateIns,
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
        numberOfRoundsToEvaluateOver: int = 2
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
        self.numberOfRoundsToEvaluateOver = numberOfRoundsToEvaluateOver
        self.parameter_rounds = [parameters_to_ndarrays(initial_parameters)]

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
        # We are actually transmitting the new weights of each of the clients and then calculating the gradient 
        # on the server side, compared to what the algorithm says in the paper it will have the same effect and 
        # allow this strategy to work with the standard clients for training models in comparison to needing to
        # change them to return the update rather than the raw weights
        # Convert results
        update_results = [
            ([
            x - y
             for (x,y) in zip(parameters_to_ndarrays(self.initial_parameters), parameters_to_ndarrays(fit_res.parameters))
            ], 1) for _, fit_res in results
        ]

        aggregated_update = aggregate(update_results)
        squared_norm_aggregated_update = self.compute_norm_squared(aggregated_update)
        assert (
            self.initial_parameters is not None
        ), "When using server-side optimization, model needs to be initialized."
        initial_weights = parameters_to_ndarrays(self.initial_parameters)


        server_learning_rate = 0
        M = len(update_results)
        for (update, _) in update_results:
            squared_norm_update = self.compute_norm_squared(update)
            server_learning_rate += squared_norm_update
        server_learning_rate /= (2 * M * (squared_norm_aggregated_update + self.ε))
        server_learning_rate = max(1, server_learning_rate)

        # SGD
        fedexp_result = [
            x - server_learning_rate * y
            for x, y in zip(initial_weights, aggregated_update)
        ]
        # Update current weights
        self.initial_parameters = ndarrays_to_parameters(fedexp_result)

        # Add the current set of parameters onto the list to be used in evaluation
        self.parameter_rounds.append(deepcopy(fedexp_result))
        if len(self.parameter_rounds) >= self.numberOfRoundsToEvaluateOver:
            # We want to remove all the elements off the front which are not in the
            # last number of rounds specified
            self.parameter_rounds = self.parameter_rounds[-self.numberOfRoundsToEvaluateOver:]

        parameters_aggregated = ndarrays_to_parameters(fedexp_result)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Also add the server learning rate to the metrics which are being recorded
        metrics_aggregated['server_learning_rate'] = server_learning_rate
        return parameters_aggregated, metrics_aggregated
    

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        
        # Use the average of the past number of specified rounds (or fewer if not that many rounds have passed)
        # during the evaluation of the model
        parameters_ndarrays = aggregate([(params, 1) for params in self.parameter_rounds])
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)

        # The parameters handed to the evaluation function should be the average of the
        # parameters stored over the past number of rounds
        average_parameters = ndarrays_to_parameters(aggregate([(params, 1) for params in self.parameter_rounds]))
        evaluate_ins = EvaluateIns(average_parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

