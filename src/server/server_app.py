"""
server_app.py -- Federated Learning Server with FedProx Strategy & Async Timeouts

Implements a fault-tolerant aggregation server that:
  1. Uses FedProx strategy (extends FedAvg) to handle extreme Non-IID data
  2. Configures 60-second round timeouts for straggler/dropout resilience
  3. Aggregates partial results when clients drop or timeout
  4. Logs per-round metrics (loss, accuracy, dropped clients, round time)
     to a structured server.log file

The server requires a minimum of 2 clients to begin training. If a client
times out or drops during a round, the server aggregates whatever results
it has received and continues to the next round.

Environment Variables:
  NUM_ROUNDS       : Total federated rounds (default: 20)
  MIN_FIT_CLIENTS  : Minimum clients needed to aggregate (default: 2)
  MIN_AVAIL_CLIENTS: Minimum clients to start a round (default: 2)
  FRACTION_FIT     : Fraction of clients sampled per round (default: 1.0)
  MU               : FedProx proximal coefficient sent to clients (default: 0.1)
  LOCAL_EPOCHS     : Local epochs per client per round (default: 3)
  LEARNING_RATE    : Client learning rate (default: 0.01)
  SERVER_PORT      : gRPC listen port (default: 8080)
  ROUND_TIMEOUT    : Seconds to wait for client responses (default: 60.0)
  LOG_DIR          : Directory for log files (default: "logs")

Usage:
    python -m src.server.server_app
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

# ============================================================
# Add project root to Python path for imports
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.resnet import get_model
from utils.logger import get_server_logger

# ============================================================
# Configuration from environment variables
# ============================================================
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "20"))
MIN_FIT_CLIENTS = int(os.environ.get("MIN_FIT_CLIENTS", "2"))
MIN_AVAIL_CLIENTS = int(os.environ.get("MIN_AVAIL_CLIENTS", "2"))
FRACTION_FIT = float(os.environ.get("FRACTION_FIT", "1.0"))
MU = float(os.environ.get("MU", "0.1"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.01"))
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8080"))
ROUND_TIMEOUT = float(os.environ.get("ROUND_TIMEOUT", "300.0"))
LOG_DIR = os.environ.get("LOG_DIR", "logs")

# ============================================================
# Initialize logger
# ============================================================
logger = get_server_logger(log_dir=LOG_DIR)


# ============================================================
# Initial Model Parameters
# ============================================================

def get_initial_parameters() -> Parameters:
    """
    Generate initial model parameters from a freshly created MedResNet18.

    These parameters are broadcast to all clients at the start of
    the first federated round, ensuring all hospitals begin from
    the same initialization.

    Returns:
        Flower Parameters object containing the initial model weights.
    """
    model = get_model()
    ndarrays = [
        val.cpu().numpy()
        for _, val in model.state_dict().items()
    ]
    return ndarrays_to_parameters(ndarrays)


# ============================================================
# Custom FedProx Strategy
# ============================================================

class FedProxStrategy(fl.server.strategy.FedAvg):
    """
    Custom FedProx aggregation strategy extending Flower's FedAvg.

    FedProx modifies federated learning to handle:
      1. Non-IID data: Sends the proximal coefficient (mu) to clients,
         which add a proximal penalty term to their local loss function
         to prevent excessive drift from the global model.
      2. Partial participation: Gracefully handles client dropouts and
         timeouts by aggregating whatever results are available.

    The strategy logs detailed per-round metrics including loss,
    accuracy, number of responding/failing clients, and round duration.
    """

    def __init__(
        self,
        mu: float = 0.1,
        local_epochs: int = 3,
        lr: float = 0.01,
        **kwargs,
    ):
        """
        Initialize the FedProx strategy.

        Args:
            mu: Proximal coefficient for FedProx (sent to clients).
            local_epochs: Number of local training epochs per round.
            lr: Learning rate for client-side SGD.
            **kwargs: Additional arguments passed to FedAvg.
        """
        super().__init__(**kwargs)
        self.mu = mu
        self.local_epochs = local_epochs
        self.lr = lr

        # --------------------------------------------------------
        # Track per-round timing for performance logging
        # --------------------------------------------------------
        self._round_start_time = None

        logger.info(
            f"FedProxStrategy initialized: mu={mu}, "
            f"local_epochs={local_epochs}, lr={lr}"
        )

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """
        Provide initial global model parameters.

        Called once at the start of federated training to initialize
        the global model that will be broadcast to all clients.

        Args:
            client_manager: Flower client manager (unused here).

        Returns:
            Initial model parameters.
        """
        logger.info("Initializing global model parameters (MedResNet18)...")
        initial_params = get_initial_parameters()
        logger.info("Global model initialized successfully.")
        return initial_params

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the training instructions sent to each client.

        Injects FedProx-specific configuration (mu, local_epochs, lr)
        into the fit instructions so clients can apply the proximal
        term during local training.

        Args:
            server_round: Current federated round number.
            parameters: Current global model parameters.
            client_manager: Flower client manager for client selection.

        Returns:
            List of (client, fit_instructions) tuples.
        """
        # --------------------------------------------------------
        # Record round start time for duration tracking
        # --------------------------------------------------------
        self._round_start_time = time.time()

        logger.info(f"=== ROUND {server_round} STARTING ===")

        # --------------------------------------------------------
        # Build FedProx-specific configuration for clients
        # --------------------------------------------------------
        config = {
            "current_round": server_round,
            "mu": self.mu,
            "local_epochs": self.local_epochs,
            "lr": self.lr,
        }

        # --------------------------------------------------------
        # Use parent's client sampling logic
        # --------------------------------------------------------
        fit_ins = FitIns(parameters, config)

        # Sample clients using the parent's fraction_fit logic
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        logger.info(
            f"Round {server_round}: Sampled {len(clients)} clients "
            f"(min_required={min_num_clients})"
        )

        # --------------------------------------------------------
        # Send same config to all sampled clients
        # --------------------------------------------------------
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates from clients, handling failures gracefully.

        This method:
          1. Logs any client failures (timeouts, dropouts, errors)
          2. Filters out clients that returned 0 samples (dropout simulation)
          3. Aggregates remaining results using weighted FedAvg
          4. Logs comprehensive round metrics

        Args:
            server_round: Current federated round number.
            results: List of successful (client, result) tuples.
            failures: List of failed clients or exceptions.

        Returns:
            Tuple of (aggregated parameters, metrics dict).
            Returns (None, {}) if no valid results are available.
        """
        # --------------------------------------------------------
        # Calculate round duration
        # --------------------------------------------------------
        round_duration = 0.0
        if self._round_start_time is not None:
            round_duration = time.time() - self._round_start_time

        # --------------------------------------------------------
        # Log failures (timeouts, disconnections, errors)
        # --------------------------------------------------------
        num_failures = len(failures)
        if num_failures > 0:
            logger.warning(
                f"Round {server_round}: {num_failures} client failure(s) "
                f"(timeout/disconnect/error)"
            )
            for i, failure in enumerate(failures):
                if isinstance(failure, BaseException):
                    logger.warning(
                        f"  Failure {i + 1}: {type(failure).__name__}: {failure}"
                    )
                else:
                    client_proxy, fit_res = failure
                    logger.warning(
                        f"  Failure {i + 1}: Client {client_proxy.cid} "
                        f"returned error status"
                    )

        # --------------------------------------------------------
        # Filter out clients that returned 0 samples (dropped out)
        # --------------------------------------------------------
        valid_results = []
        dropped_clients = 0

        for client_proxy, fit_res in results:
            if fit_res.num_examples == 0:
                # Client simulated a dropout or encountered an error
                dropped_clients += 1
                client_metrics = fit_res.metrics or {}
                status = client_metrics.get("status", "unknown")
                hospital = client_metrics.get("hospital_id", "unknown")
                logger.warning(
                    f"Round {server_round}: Client {hospital} returned "
                    f"0 samples (status={status}). Excluding from aggregation."
                )
            else:
                valid_results.append((client_proxy, fit_res))

        # --------------------------------------------------------
        # Check if we have enough valid results to aggregate
        # --------------------------------------------------------
        total_responding = len(valid_results)
        total_dropped = dropped_clients + num_failures

        if not valid_results:
            logger.error(
                f"Round {server_round}: No valid results to aggregate! "
                f"({total_dropped} client(s) dropped/failed). "
                f"Skipping aggregation."
            )
            # ------------------------------------------------
            # Log round summary even on failure
            # ------------------------------------------------
            logger.info(
                f"=== ROUND {server_round} SUMMARY === "
                f"responding={total_responding}, "
                f"dropped={total_dropped}, "
                f"duration={round_duration:.1f}s, "
                f"status=SKIPPED (no valid results)"
            )
            return None, {}

        # --------------------------------------------------------
        # Aggregate using parent FedAvg weighted averaging
        # --------------------------------------------------------
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, valid_results, failures
        )

        # --------------------------------------------------------
        # Collect per-client metrics for logging
        # --------------------------------------------------------
        client_losses = []
        for _, fit_res in valid_results:
            client_metrics = fit_res.metrics or {}
            loss = client_metrics.get("loss", None)
            if loss is not None:
                client_losses.append(float(loss))

        avg_loss = np.mean(client_losses) if client_losses else float("nan")

        # --------------------------------------------------------
        # Log comprehensive round summary
        # --------------------------------------------------------
        logger.info(
            f"=== ROUND {server_round} SUMMARY === "
            f"avg_loss={avg_loss:.4f}, "
            f"responding={total_responding}, "
            f"dropped={total_dropped}, "
            f"failures={num_failures}, "
            f"duration={round_duration:.1f}s"
        )

        # --------------------------------------------------------
        # Return aggregated parameters and server-side metrics
        # --------------------------------------------------------
        metrics = {
            "avg_loss": float(avg_loss),
            "responding_clients": total_responding,
            "dropped_clients": total_dropped,
            "failures": num_failures,
            "round_duration": round_duration,
        }

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.

        Computes weighted average loss and accuracy across all
        responding clients.

        Args:
            server_round: Current federated round number.
            results: List of successful evaluation results.
            failures: List of evaluation failures.

        Returns:
            Tuple of (weighted average loss, metrics dict).
        """
        if not results:
            logger.warning(
                f"Round {server_round}: No evaluation results received."
            )
            return None, {}

        # --------------------------------------------------------
        # Compute weighted average loss and accuracy
        # --------------------------------------------------------
        total_samples = 0
        weighted_loss = 0.0
        weighted_accuracy = 0.0

        for _, evaluate_res in results:
            num_samples = evaluate_res.num_examples
            total_samples += num_samples
            weighted_loss += evaluate_res.loss * num_samples

            client_metrics = evaluate_res.metrics or {}
            accuracy = client_metrics.get("accuracy", 0.0)
            weighted_accuracy += accuracy * num_samples

        avg_loss = weighted_loss / max(total_samples, 1)
        avg_accuracy = weighted_accuracy / max(total_samples, 1)

        logger.info(
            f"Round {server_round} EVALUATION: "
            f"avg_loss={avg_loss:.4f}, avg_accuracy={avg_accuracy:.4f}, "
            f"total_samples={total_samples}, "
            f"eval_failures={len(failures)}"
        )

        return avg_loss, {
            "avg_accuracy": avg_accuracy,
            "total_samples": total_samples,
        }


# ============================================================
# Server Startup
# ============================================================

def start_server():
    """
    Initialize and start the Flower federated learning server.

    Creates the FedProx strategy with configured parameters, sets
    up async timeouts, and starts the gRPC server.
    """
    logger.info("=" * 60)
    logger.info("Starting Federated Learning Server")
    logger.info(f"  Rounds:          {NUM_ROUNDS}")
    logger.info(f"  Min Fit Clients: {MIN_FIT_CLIENTS}")
    logger.info(f"  Min Available:   {MIN_AVAIL_CLIENTS}")
    logger.info(f"  Fraction Fit:    {FRACTION_FIT}")
    logger.info(f"  FedProx mu:      {MU}")
    logger.info(f"  Local Epochs:    {LOCAL_EPOCHS}")
    logger.info(f"  Learning Rate:   {LEARNING_RATE}")
    logger.info(f"  Round Timeout:   {ROUND_TIMEOUT}s")
    logger.info(f"  Server Port:     {SERVER_PORT}")
    logger.info("=" * 60)

    # --------------------------------------------------------
    # Create FedProx strategy
    # --------------------------------------------------------
    strategy = FedProxStrategy(
        mu=MU,
        local_epochs=LOCAL_EPOCHS,
        lr=LEARNING_RATE,
        fraction_fit=FRACTION_FIT,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_available_clients=MIN_AVAIL_CLIENTS,
        # Accept partial results if some clients fail
        accept_failures=True,
    )

    # --------------------------------------------------------
    # Configure server with round timeout
    # The server will wait up to ROUND_TIMEOUT seconds for
    # client responses before proceeding with whatever results
    # it has. This handles our simulated stragglers and dropouts.
    # --------------------------------------------------------
    server_config = fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS,
        round_timeout=ROUND_TIMEOUT,
    )

    # --------------------------------------------------------
    # Start the gRPC server
    # --------------------------------------------------------
    server_address = f"0.0.0.0:{SERVER_PORT}"
    logger.info(f"Listening on {server_address}...")

    fl.server.start_server(
        server_address=server_address,
        config=server_config,
        strategy=strategy,
    )

    logger.info("Server shut down. Federated training complete.")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    start_server()
