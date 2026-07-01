"""
client_app.py -- Federated Learning Client with Differential Privacy & Chaos Engineering

This module implements a Flower client for a hospital node in the federated
learning network. Each client:
  1. Loads its Non-IID partition of BloodMNIST based on HOSPITAL_ID env var
  2. Trains a MedResNet18 with Opacus DP-SGD (epsilon configurable via env var)
  3. Applies FedProx proximal term to mitigate client drift from Non-IID data
  4. Simulates real-world infrastructure failures:
     - Straggler simulation: random 10-45 second delays
     - Dropout simulation: graceful per-round failure (process stays alive)

Environment Variables:
  HOSPITAL_ID      : "hospital_a" or "hospital_b" (required)
  SERVER_ADDRESS   : Flower server address (default: "127.0.0.1:8080")
  EPSILON          : Differential privacy budget (default: 10.0)
  MU               : FedProx proximal coefficient (default: 0.1)
  LOCAL_EPOCHS     : Number of local training epochs per round (default: 3)
  BATCH_SIZE       : Training batch size (default: 32)
  LEARNING_RATE    : SGD learning rate (default: 0.01)
  STRAGGLER_PROB   : Probability of straggler delay per round (default: 0.3)
  DROPOUT_PROB     : Probability of client dropout per round (default: 0.15)
  DATA_DIR         : Root directory for dataset (default: "./data")
  MAX_GRAD_NORM    : DP-SGD per-sample gradient clipping bound (default: 1.0)
  DP_DELTA         : DP delta parameter (default: 1e-5)

Usage:
    # As Hospital A
    HOSPITAL_ID=hospital_a python -m src.client.client_app

    # As Hospital B with custom privacy budget
    HOSPITAL_ID=hospital_b EPSILON=5.0 python -m src.client.client_app
"""

import os
import sys
import time
import random
import copy
from collections import OrderedDict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import NDArrays, Scalar

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

# ============================================================
# Add project root to Python path for imports
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.resnet import get_model
from utils.generate_non_iid import load_hospital_data
from utils.logger import get_client_logger

# ============================================================
# Configuration from environment variables
# ============================================================
HOSPITAL_ID = os.environ.get("HOSPITAL_ID", "hospital_a")
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "127.0.0.1:8080")

# Privacy budget (epsilon) — configurable via env var, defaults to 10.0
EPSILON = float(os.environ.get("EPSILON", "10.0"))

# FedProx proximal coefficient
MU = float(os.environ.get("MU", "0.1"))

# Training hyperparameters
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.01"))

# Chaos engineering probabilities
STRAGGLER_PROB = float(os.environ.get("STRAGGLER_PROB", "0.3"))
DROPOUT_PROB = float(os.environ.get("DROPOUT_PROB", "0.15"))

# Data and DP configuration
DATA_DIR = os.environ.get("DATA_DIR", "./data")
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
DP_DELTA = float(os.environ.get("DP_DELTA", "1e-5"))
MAX_PHYSICAL_BATCH_SIZE = int(os.environ.get("MAX_PHYSICAL_BATCH_SIZE", "16"))

# ============================================================
# Device configuration
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Initialize logger
# ============================================================
logger = get_client_logger(HOSPITAL_ID)


# ============================================================
# Chaos Engineering Functions
# ============================================================

def simulate_straggler() -> float:
    """
    Simulate a network straggler by sleeping for a random duration.

    With probability STRAGGLER_PROB, the client will sleep for a
    random duration between 10 and 45 seconds before returning its
    model weights. This simulates slow hospital Wi-Fi or overloaded
    network infrastructure.

    Returns:
        The number of seconds slept (0.0 if no straggler event).
    """
    if random.random() < STRAGGLER_PROB:
        delay = random.uniform(10, 45)
        logger.warning(
            f"STRAGGLER SIMULATION: Sleeping {delay:.1f}s "
            f"(simulating slow network for {HOSPITAL_ID})"
        )
        time.sleep(delay)
        return delay
    return 0.0


def simulate_dropout() -> bool:
    """
    Simulate a client dropout / disconnection event.

    With probability DROPOUT_PROB, this function returns True,
    indicating the client should skip this round. The client does
    NOT crash or terminate — it simply fails to return weights for
    this specific round and remains alive for the next one.

    This is implemented as a boolean check rather than an exception
    to prevent the Docker container from terminating.

    Returns:
        True if the client should drop out this round, False otherwise.
    """
    if random.random() < DROPOUT_PROB:
        logger.error(
            f"DROPOUT SIMULATION: {HOSPITAL_ID} dropping out this round! "
            f"Client will skip training but remain alive for next round."
        )
        return True
    return False


# ============================================================
# Training and Evaluation Functions
# ============================================================

def train_one_round(
    model: nn.Module,
    train_loader: DataLoader,
    global_params: List[torch.Tensor],
    mu: float,
    local_epochs: int,
    lr: float,
    epsilon: float,
    delta: float,
    max_grad_norm: float,
) -> Tuple[float, int]:
    """
    Train the model locally for one federated round with DP-SGD and FedProx.

    This function:
      1. Wraps the model/optimizer/loader with Opacus DP-SGD
      2. Trains for `local_epochs` with the FedProx proximal term
      3. Tracks and reports the privacy budget spent

    FedProx + Opacus Interaction:
      The FedProx proximal gradient is injected AFTER loss.backward()
      but BEFORE optimizer.step(). This is critical because:
        - loss.backward() must only see the data-dependent cross-entropy
          loss so Opacus can cleanly compute per-sample gradients and
          clip them individually.
        - The proximal gradient d/dw[(mu/2)||w-w*||^2] = mu*(w-w*) is
          NOT data-dependent, so it does not need per-sample clipping.
          We add it directly to .grad before the optimizer step.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for this hospital's training data.
        global_params: List of global model parameter tensors (for FedProx).
        mu: FedProx proximal coefficient.
        local_epochs: Number of local training epochs.
        lr: Learning rate.
        epsilon: Target differential privacy budget.
        delta: DP delta parameter.
        max_grad_norm: Per-sample gradient clipping bound.

    Returns:
        Tuple of (average loss over all batches, total samples trained on).
    """
    model.train()
    model.to(DEVICE)

    # --------------------------------------------------------
    # Set up optimizer
    # --------------------------------------------------------
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Wrap with Opacus DP-SGD Privacy Engine
    # --------------------------------------------------------
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=epsilon,
        target_delta=delta,
        epochs=local_epochs,
        max_grad_norm=max_grad_norm,
    )

    logger.info(
        f"Opacus DP-SGD initialized: target_epsilon={epsilon}, "
        f"delta={delta}, max_grad_norm={max_grad_norm}"
    )

    # --------------------------------------------------------
    # Move global params to device for FedProx computation
    # --------------------------------------------------------
    global_params_device = [p.to(DEVICE) for p in global_params]

    # --------------------------------------------------------
    # Local training loop with FedProx proximal term
    # --------------------------------------------------------
    # BatchMemoryManager splits large logical batches into smaller
    # physical micro-batches for memory efficiency. This is critical
    # for Opacus on CPU — per-sample gradients require O(batch_size)
    # memory multiplier. Smaller physical batches keep memory bounded
    # while maintaining the same privacy accounting.
    # --------------------------------------------------------
    total_loss = 0.0
    total_samples = 0

    for epoch in range(local_epochs):
        epoch_loss = 0.0
        epoch_samples = 0

        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer,
        ) as memory_safe_loader:
            for batch_idx, (data, target) in enumerate(memory_safe_loader):
                data = data.to(DEVICE)
                # BloodMNIST labels are shape (batch, 1), squeeze to (batch,)
                target = target.squeeze().long().to(DEVICE)

                optimizer.zero_grad()

                # Forward pass — data-dependent loss ONLY
                output = model(data)
                loss = criterion(output, target)

                # ------------------------------------------------
                # Backward pass — Opacus computes per-sample
                # gradients and clips them. Only the data-dependent
                # cross-entropy loss goes through backward() so
                # Opacus's hooks can work cleanly without conflicts.
                # ------------------------------------------------
                loss.backward()

                # ------------------------------------------------
                # FedProx proximal gradient injection (post-backward)
                #
                # The proximal term gradient:
                #   d/dw [(mu/2) * ||w - w_global||^2] = mu * (w - w_global)
                #
                # This is NOT data-dependent (it only depends on model
                # weights, not individual patient images), so it does
                # not need per-sample DP clipping. We inject it directly
                # into .grad after Opacus has finished its per-sample
                # clipping, but before optimizer.step() applies the update.
                # ------------------------------------------------
                with torch.no_grad():
                    for local_w, global_w in zip(
                        model.parameters(), global_params_device
                    ):
                        if local_w.grad is not None:
                            local_w.grad.add_(
                                mu * (local_w.data - global_w.data)
                            )

                optimizer.step()

                batch_size = data.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size

        avg_epoch_loss = epoch_loss / max(epoch_samples, 1)
        logger.info(
            f"  Epoch {epoch + 1}/{local_epochs}: "
            f"loss={avg_epoch_loss:.4f}, samples={epoch_samples}"
        )

        total_loss += epoch_loss
        total_samples += epoch_samples

    # --------------------------------------------------------
    # Report final privacy budget spent
    # --------------------------------------------------------
    final_epsilon = privacy_engine.get_epsilon(delta=delta)
    logger.info(
        f"Privacy budget spent this round: epsilon={final_epsilon:.2f} "
        f"(target={epsilon})"
    )

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, total_samples


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
) -> Tuple[float, float, int]:
    """
    Evaluate the model on the hospital's test partition.

    Args:
        model: The trained model to evaluate.
        test_loader: DataLoader for this hospital's test data.

    Returns:
        Tuple of (average loss, accuracy, total test samples).
    """
    model.eval()
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(DEVICE)
            target = target.squeeze().long().to(DEVICE)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += data.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy, total


# ============================================================
# Flower Client Implementation
# ============================================================

class HospitalClient(fl.client.NumPyClient):
    """
    Flower NumPy client representing a single hospital in the federation.

    Each hospital client:
      - Loads its Non-IID data partition (healthy or diseased cells)
      - Trains locally with Opacus DP-SGD for gradient-level privacy
      - Applies FedProx proximal term to combat Non-IID client drift
      - Simulates real-world infrastructure chaos (stragglers, dropouts)

    The client gracefully handles dropout events without crashing,
    allowing the Docker container to stay alive across rounds.
    """

    def __init__(
        self,
        hospital_id: str,
        data_dir: str = "./data",
    ):
        """
        Initialize the hospital client.

        Args:
            hospital_id: Hospital identifier ("hospital_a" or "hospital_b").
            data_dir: Root directory for the dataset.
        """
        super().__init__()
        self.hospital_id = hospital_id
        self.data_dir = data_dir

        # --------------------------------------------------------
        # Initialize model
        # --------------------------------------------------------
        self.model = get_model()
        self.model.to(DEVICE)

        logger.info(
            f"Initialized {hospital_id} client on device={DEVICE}"
        )

        # --------------------------------------------------------
        # Load hospital-specific data partitions
        # --------------------------------------------------------
        logger.info(f"Loading data partition for {hospital_id}...")

        train_subset = load_hospital_data(hospital_id, data_dir, split="train")
        test_subset = load_hospital_data(hospital_id, data_dir, split="test")

        self.train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,   # Required by Opacus for uniform batch sizes
            num_workers=0,    # Safe default for Docker containers
        )
        self.test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        logger.info(
            f"Data loaded: {len(train_subset)} train samples, "
            f"{len(test_subset)} test samples"
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Return model parameters as a list of NumPy arrays.

        Args:
            config: Configuration dictionary from the server (unused here).

        Returns:
            List of NumPy arrays representing model weights.
        """
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model parameters from a list of NumPy arrays.

        Loads the global model weights received from the server
        into the local model.

        Args:
            parameters: List of NumPy arrays from the server.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model locally for one federated round.

        This method:
          1. Checks for simulated dropout (skips round if triggered)
          2. Loads global model weights
          3. Simulates straggler delay
          4. Trains with DP-SGD + FedProx
          5. Returns updated weights

        The dropout simulation is caught gracefully — if a dropout
        event occurs, the method returns the unchanged global weights
        with zero samples, signaling to the server that this client
        did not contribute this round. The client process stays alive.

        Args:
            parameters: Global model parameters from the server.
            config: Round configuration from the server strategy.

        Returns:
            Tuple of (updated parameters, num_samples, metrics dict).
        """
        # --------------------------------------------------------
        # Read round-specific config from server
        # --------------------------------------------------------
        current_round = config.get("current_round", 0)
        mu = config.get("mu", MU)
        local_epochs = config.get("local_epochs", LOCAL_EPOCHS)
        lr = config.get("lr", LEARNING_RATE)

        logger.info(
            f"--- Round {current_round} START --- "
            f"(mu={mu}, epochs={local_epochs}, lr={lr})"
        )

        # --------------------------------------------------------
        # Simulate client dropout (graceful, no crash)
        # --------------------------------------------------------
        try:
            if simulate_dropout():
                # Return unchanged global weights with 0 samples
                # This tells the server: "I didn't train this round"
                logger.warning(
                    f"Round {current_round}: Dropout triggered. "
                    f"Returning unchanged weights."
                )
                return parameters, 0, {
                    "status": "dropped",
                    "hospital_id": self.hospital_id,
                    "round": current_round,
                }

            # --------------------------------------------------------
            # Load global model weights
            # --------------------------------------------------------
            self.set_parameters(parameters)

            # Save a copy of global params for FedProx proximal term
            global_params = [
                p.clone().detach()
                for p in self.model.parameters()
            ]

            # --------------------------------------------------------
            # Simulate straggler delay
            # --------------------------------------------------------
            straggler_delay = simulate_straggler()

            # --------------------------------------------------------
            # Create a fresh model copy for Opacus
            # (Opacus wraps the model and cannot be reused across rounds)
            # --------------------------------------------------------
            train_model = get_model()
            train_model.load_state_dict(self.model.state_dict())

            # --------------------------------------------------------
            # Train locally with DP-SGD + FedProx
            # --------------------------------------------------------
            avg_loss, num_samples = train_one_round(
                model=train_model,
                train_loader=self.train_loader,
                global_params=global_params,
                mu=mu,
                local_epochs=local_epochs,
                lr=lr,
                epsilon=EPSILON,
                delta=DP_DELTA,
                max_grad_norm=MAX_GRAD_NORM,
            )

            # --------------------------------------------------------
            # Copy trained weights back to the main model
            # --------------------------------------------------------
            # Opacus wraps the model with _module, extract the actual state dict
            if hasattr(train_model, '_module'):
                trained_state = train_model._module.state_dict()
            else:
                trained_state = train_model.state_dict()
            self.model.load_state_dict(trained_state)

            # --------------------------------------------------------
            # Return updated parameters
            # --------------------------------------------------------
            updated_params = self.get_parameters(config={})

            metrics = {
                "loss": float(avg_loss),
                "hospital_id": self.hospital_id,
                "round": current_round,
                "straggler_delay": float(straggler_delay),
                "epsilon": float(EPSILON),
                "status": "trained",
            }

            logger.info(
                f"--- Round {current_round} COMPLETE --- "
                f"loss={avg_loss:.4f}, samples={num_samples}, "
                f"straggler_delay={straggler_delay:.1f}s"
            )

            return updated_params, num_samples, metrics

        except Exception as e:
            # --------------------------------------------------------
            # Catch ALL exceptions to prevent container crash.
            # The client fails this round but stays alive for the next.
            # --------------------------------------------------------
            logger.error(
                f"Round {current_round}: Exception during training: {e}. "
                f"Returning unchanged weights. Client remains alive."
            )
            return parameters, 0, {
                "status": "error",
                "error": str(e),
                "hospital_id": self.hospital_id,
                "round": current_round,
            }

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the global model on this hospital's test partition.

        Args:
            parameters: Global model parameters from the server.
            config: Evaluation configuration from the server.

        Returns:
            Tuple of (loss, num_samples, metrics dict).
        """
        self.set_parameters(parameters)

        loss, accuracy, num_samples = evaluate_model(
            self.model, self.test_loader
        )

        logger.info(
            f"Evaluation: loss={loss:.4f}, accuracy={accuracy:.4f}, "
            f"samples={num_samples}"
        )

        return float(loss), num_samples, {
            "accuracy": float(accuracy),
            "hospital_id": self.hospital_id,
        }


# ============================================================
# Client Startup
# ============================================================

def start_client():
    """
    Initialize and start the Flower client.

    Reads the HOSPITAL_ID environment variable to determine which
    data partition to load, creates the HospitalClient, and connects
    to the Flower server.
    """
    logger.info("=" * 60)
    logger.info(f"Starting Federated Learning Client")
    logger.info(f"  Hospital ID:    {HOSPITAL_ID}")
    logger.info(f"  Server:         {SERVER_ADDRESS}")
    logger.info(f"  Privacy (eps):  {EPSILON}")
    logger.info(f"  FedProx (mu):   {MU}")
    logger.info(f"  Local Epochs:   {LOCAL_EPOCHS}")
    logger.info(f"  Batch Size:     {BATCH_SIZE}")
    logger.info(f"  Learning Rate:  {LEARNING_RATE}")
    logger.info(f"  Straggler Prob: {STRAGGLER_PROB}")
    logger.info(f"  Dropout Prob:   {DROPOUT_PROB}")
    logger.info(f"  Device:         {DEVICE}")
    logger.info("=" * 60)

    # --------------------------------------------------------
    # Create the hospital client
    # --------------------------------------------------------
    client = HospitalClient(
        hospital_id=HOSPITAL_ID,
        data_dir=DATA_DIR,
    )

    # --------------------------------------------------------
    # Connect to the Flower server
    # --------------------------------------------------------
    logger.info(f"Connecting to server at {SERVER_ADDRESS}...")

    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
    )

    logger.info("Client disconnected from server. Shutting down.")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    start_client()
