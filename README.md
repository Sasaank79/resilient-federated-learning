# Resilient & Private Federated Learning Network

A production-grade federated learning system for medical image classification that handles **extreme Non-IID data skew**, **differential privacy**, **network failures**, and **containerized deployment** — built with PyTorch, Opacus, Flower, and Docker.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Compose Network                      │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │  hospital_a_net   │    (internal, isolated)                  │
│  │                    │                                          │
│  │  ┌──────────────┐ │         ┌──────────────────┐             │
│  │  │ Hospital A    │ │  gRPC   │   Aggregator     │             │
│  │  │ (Healthy      │◄────────►│                  │             │
│  │  │  Classes 0-3) │ │  :8080  │  FedProx Strategy│             │
│  │  │              │ │         │  60s Timeout     │             │
│  │  │  Opacus DP   │ │         │  Structured Logs │             │
│  │  │  FedProx     │ │         │                  │             │
│  │  │  Chaos Sim   │ │         └────────┬─────────┘             │
│  │  └──────────────┘ │                  │                       │
│  └──────────────────┘                  │                       │
│                                         │                       │
│  ┌──────────────────┐                  │                       │
│  │  hospital_b_net   │    (internal, isolated)                  │
│  │                    │                  │                       │
│  │  ┌──────────────┐ │         gRPC     │                       │
│  │  │ Hospital B    │◄─────────────────┘                       │
│  │  │ (Diseased     │ │  :8080                                  │
│  │  │  Classes 4-7) │ │                                          │
│  │  │              │ │   Hospitals CANNOT                       │
│  │  │  Opacus DP   │ │   communicate with                      │
│  │  │  FedProx     │ │   each other.                           │
│  │  │  Chaos Sim   │ │                                          │
│  │  └──────────────┘ │                                          │
│  └──────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### Privacy-Preserving Training (Opacus DP-SGD)
- **Differential Privacy** injected at the gradient level via [Opacus](https://opacus.ai/)
- Privacy budget (epsilon) configurable via environment variable (default: 10.0)
- Even intercepted model updates cannot reverse-engineer patient images
- Per-sample gradient clipping with calibrated Gaussian noise

### Extreme Non-IID Data Partitioning
- **Hospital A** receives ONLY healthy blood cell types (Basophil, Eosinophil, Erythroblast, Immature Granulocytes)
- **Hospital B** receives ONLY diseased/pathological types (Lymphocyte, Monocyte, Neutrophil, Platelet)
- **Zero class overlap** between hospitals — mathematically verified
- Uses [BloodMNIST](https://medmnist.com/) (17,092 real medical images, 28x28 RGB, 8 classes)

### FedProx Aggregation (Non-IID Robust)
- Proximal term `(mu/2) * ||w - w_global||^2` prevents client drift
- Handles partial participation — aggregates whatever results arrive
- Outperforms standard FedAvg on heterogeneous data distributions

### Chaos Engineering
- **Straggler Simulation**: Clients randomly sleep 10-45 seconds (30% probability)
- **Dropout Simulation**: Clients gracefully skip rounds (15% probability) without crashing
- **60-second round timeout**: Server proceeds with partial results after timeout

### Production Infrastructure
- **Docker Compose** one-click deployment with 3 isolated services
- **Network isolation**: Hospitals communicate ONLY with the aggregator via gRPC
- **Structured logging**: Rotating file logs (10MB, 5 backups) with per-round metrics
- **Health checks** on all containers

---

## Project Structure

```
resilient-federated-healthcare/
│
├── data/                           # Downloaded datasets (gitignored)
├── logs/                           # Server & client logs (gitignored)
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── resnet.py               # Modified ResNet-18 (GroupNorm, Dropout)
│   ├── server/
│   │   ├── __init__.py
│   │   ├── server_app.py           # FedProx strategy & async timeouts
│   │   └── Dockerfile.server
│   └── client/
│       ├── __init__.py
│       ├── client_app.py           # Opacus DP-SGD & chaos engineering
│       └── Dockerfile.client
│
├── utils/
│   ├── __init__.py
│   ├── generate_non_iid.py         # Extreme Non-IID BloodMNIST partitioner
│   └── logger.py                   # Structured rotating file logger
│
├── docker-compose.yml              # One-click isolated infrastructure
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Sasaank79/resilient-federated-learning.git
cd resilient-federated-learning

# Build and launch all services
docker-compose build
docker-compose up

# View logs
docker-compose logs -f aggregator
docker-compose logs -f hospital_a
docker-compose logs -f hospital_b

# Shut down
docker-compose down
```

### Option 2: Local Development

```bash
# Clone and install dependencies
git clone https://github.com/Sasaank79/resilient-federated-learning.git
cd resilient-federated-learning
pip install -r requirements.txt

# Step 1: Download and partition the dataset
python -m utils.generate_non_iid --data_dir ./data

# Step 2: Start the aggregation server (Terminal 1)
python -m src.server.server_app

# Step 3: Start Hospital A client (Terminal 2)
set HOSPITAL_ID=hospital_a
python -m src.client.client_app

# Step 4: Start Hospital B client (Terminal 3)
set HOSPITAL_ID=hospital_b
python -m src.client.client_app
```

---

## Configuration Reference

All parameters are configurable via environment variables.

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_ROUNDS` | `20` | Total federated training rounds |
| `MIN_FIT_CLIENTS` | `2` | Minimum clients to aggregate |
| `MIN_AVAIL_CLIENTS` | `2` | Minimum clients to start a round |
| `FRACTION_FIT` | `1.0` | Fraction of clients sampled per round |
| `MU` | `0.1` | FedProx proximal coefficient |
| `LOCAL_EPOCHS` | `3` | Local training epochs per round |
| `LEARNING_RATE` | `0.01` | Client-side SGD learning rate |
| `SERVER_PORT` | `8080` | gRPC listen port |
| `ROUND_TIMEOUT` | `60.0` | Seconds to wait for client responses |
| `LOG_DIR` | `logs` | Directory for log files |

### Client Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOSPITAL_ID` | `hospital_a` | Hospital identifier (`hospital_a` or `hospital_b`) |
| `SERVER_ADDRESS` | `127.0.0.1:8080` | Flower server address |
| `EPSILON` | `10.0` | Differential privacy budget |
| `MU` | `0.1` | FedProx proximal coefficient |
| `LOCAL_EPOCHS` | `3` | Local training epochs per round |
| `BATCH_SIZE` | `32` | Training batch size |
| `LEARNING_RATE` | `0.01` | SGD learning rate |
| `STRAGGLER_PROB` | `0.3` | Probability of straggler delay (0.0-1.0) |
| `DROPOUT_PROB` | `0.15` | Probability of round dropout (0.0-1.0) |
| `DATA_DIR` | `./data` | Dataset root directory |
| `MAX_GRAD_NORM` | `1.0` | DP-SGD per-sample gradient clipping |
| `DP_DELTA` | `1e-5` | DP delta parameter |

---

## Privacy Guarantees

This system uses **Differential Privacy with Stochastic Gradient Descent (DP-SGD)** via [Opacus](https://opacus.ai/).

### How It Works
1. **Per-sample gradient clipping**: Each training sample's gradient is clipped to `MAX_GRAD_NORM`, bounding the influence of any single patient's data
2. **Calibrated noise injection**: Gaussian noise is added to the clipped gradients before aggregation
3. **Privacy accounting**: The cumulative privacy budget (epsilon) is tracked across rounds

### Privacy Budget (Epsilon)
- **epsilon = 1.0**: Strong privacy, slower convergence
- **epsilon = 10.0**: Moderate privacy (default), good utility-privacy tradeoff
- **epsilon = 50.0**: Weak privacy, near-baseline accuracy

### GroupNorm Substitution
Opacus requires per-sample gradient isolation, which is incompatible with `BatchNorm` (batch-wide statistics). All `BatchNorm` layers in ResNet-18 are automatically replaced with `GroupNorm` via `opacus.validators.ModuleValidator.fix()`.

---

## Non-IID Data Strategy

### The Problem
Standard federated learning (FedAvg) assumes each client's data is roughly identically distributed (IID). In healthcare, this is unrealistic — different hospitals see different patient populations.

### Our Approach
We create a **worst-case scenario** to stress-test the system:

| Hospital | Classes | Cell Types | Train Samples |
|----------|---------|------------|---------------|
| Hospital A | 0, 1, 2, 3 | Basophil, Eosinophil, Erythroblast, Immature Granulocytes | ~6,144 |
| Hospital B | 4, 5, 6, 7 | Lymphocyte, Monocyte, Neutrophil, Platelet | ~5,815 |

**Zero class overlap** — each hospital sees entirely different diseases. This forces FedProx to earn its keep.

### FedProx Solution
The proximal term `(mu/2) * ||w - w_global||^2` in each client's loss function penalizes local model drift from the global model, preventing the catastrophic divergence that FedAvg suffers under extreme Non-IID conditions.

---

## Monitoring & Logs

### Server Logs (`logs/server.log`)
Each round logs structured metrics:
```
[2026-06-30 15:30:00] [INFO    ] [server] === ROUND 5 SUMMARY === avg_loss=1.2345, responding=2, dropped=0, failures=0, duration=12.3s
```

### Client Logs (`logs/client.log`)
Per-round training details with privacy tracking:
```
[2026-06-30 15:30:05] [INFO    ] [client.hospital_a] Privacy budget spent this round: epsilon=3.45 (target=10.0)
[2026-06-30 15:30:10] [WARNING ] [client.hospital_b] STRAGGLER SIMULATION: Sleeping 23.4s
[2026-06-30 15:30:40] [ERROR   ] [client.hospital_a] DROPOUT SIMULATION: hospital_a dropping out this round!
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `BATCH_SIZE` or use CPU (`CUDA_VISIBLE_DEVICES=""`) |
| `No clients available` | Ensure both hospital containers are running before the server starts sampling |
| `Opacus validation error` | The model auto-fixes BatchNorm. If custom layers fail, wrap them with GroupNorm manually |
| `Connection refused` | Check that `SERVER_ADDRESS` matches the aggregator's hostname and port |
| Docker build fails | Ensure Docker Desktop is running and has sufficient memory (4GB+) |

### Verifying Network Isolation
```bash
# From hospital_a container, try to ping hospital_b (should FAIL)
docker exec fl_hospital_a ping fl_hospital_b

# From hospital_a container, try to reach aggregator (should SUCCEED)
docker exec fl_hospital_a ping fl_aggregator
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Deep Learning | PyTorch 2.0+ | CNN training & inference |
| Model | Modified ResNet-18 | Medical image classification |
| Dataset | BloodMNIST (MedMNIST) | 17K real blood cell images |
| Privacy | Opacus (DP-SGD) | Gradient-level differential privacy |
| Federation | Flower (flwr) | gRPC-based federated learning |
| Aggregation | FedProx | Non-IID robust aggregation |
| Infrastructure | Docker Compose | Containerized, isolated deployment |
| Logging | Python logging | Rotating structured file logs |

---

## License

This project is for educational and portfolio demonstration purposes.

---

## References

- [FedProx Paper](https://arxiv.org/abs/1812.06127) — Li et al., 2020
- [Opacus](https://opacus.ai/) — Differential Privacy for PyTorch
- [Flower](https://flower.ai/) — Federated Learning Framework
- [MedMNIST](https://medmnist.com/) — Medical Image Benchmarks
- [BloodMNIST](https://medmnist.com/) — Blood Cell Classification Dataset
