"""
generate_non_iid.py — Extreme Non-IID Data Partitioner for BloodMNIST

Creates a severe class-skewed partition simulating two hospitals:
  - Hospital A: Receives ONLY healthy cell types (classes 0–3)
      → Basophil, Eosinophil, Erythroblast, Immature Granulocytes
  - Hospital B: Receives ONLY diseased/pathological cell types (classes 4–7)
      → Lymphocyte, Monocyte, Neutrophil, Platelet

This partition mathematically sabotages standard (IID) training and forces
the federated system to handle extreme data heterogeneity.

Usage:
    python -m utils.generate_non_iid [--data_dir ./data]
"""

import os
import argparse
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Subset
from medmnist import BloodMNIST
from torchvision import transforms

# ============================================================
# Class-to-Hospital Mapping
# ============================================================
# Hospital A — healthy / common blood cell types
HEALTHY_CLASSES = {0, 1, 2, 3}  # Basophil, Eosinophil, Erythroblast, Immature Granulocytes

# Hospital B — diseased / pathological indicators
DISEASED_CLASSES = {4, 5, 6, 7}  # Lymphocyte, Monocyte, Neutrophil, Platelet

# Human-readable class names from BloodMNIST metadata
CLASS_NAMES = {
    0: "Basophil",
    1: "Eosinophil",
    2: "Erythroblast",
    3: "Immature Granulocytes",
    4: "Lymphocyte",
    5: "Monocyte",
    6: "Neutrophil",
    7: "Platelet",
}

# ============================================================
# Standard transforms for 28x28 RGB medical images
# ============================================================
TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def _extract_indices_by_class(
    dataset: BloodMNIST,
    target_classes: set,
) -> List[int]:
    """
    Extract sample indices belonging to the specified class set.

    Args:
        dataset: A BloodMNIST dataset instance.
        target_classes: Set of integer class labels to filter for.

    Returns:
        List of integer indices into the dataset.
    """
    indices = []
    for idx, label in enumerate(dataset.labels):
        # BloodMNIST labels are shape (1,) numpy arrays
        class_id = int(label[0])
        if class_id in target_classes:
            indices.append(idx)
    return indices


def _compute_class_distribution(
    dataset: BloodMNIST,
    indices: List[int],
) -> Dict[str, int]:
    """
    Compute per-class sample counts for a given subset of indices.

    Args:
        dataset: The full BloodMNIST dataset.
        indices: List of indices defining the subset.

    Returns:
        Dictionary mapping class names to sample counts.
    """
    labels = [int(dataset.labels[i][0]) for i in indices]
    counts = Counter(labels)
    return {CLASS_NAMES[cls]: count for cls, count in sorted(counts.items())}


def partition_dataset(
    data_dir: str = "./data",
) -> Dict[str, Dict[str, Subset]]:
    """
    Download BloodMNIST and partition into extreme Non-IID splits.

    Downloads the dataset if not already present, then creates two
    hospital partitions with zero class overlap:
      - hospital_a: classes 0–3 (healthy cells)
      - hospital_b: classes 4–7 (diseased cells)

    Each hospital gets its own train/val/test subsets.

    Args:
        data_dir: Root directory for dataset download and storage.

    Returns:
        Dictionary with structure:
        {
            "hospital_a": {"train": Subset, "val": Subset, "test": Subset},
            "hospital_b": {"train": Subset, "val": Subset, "test": Subset},
        }
    """
    os.makedirs(data_dir, exist_ok=True)

    partitions = {"hospital_a": {}, "hospital_b": {}}

    # --------------------------------------------------------
    # Process each split (train, val, test)
    # --------------------------------------------------------
    for split in ["train", "val", "test"]:
        # Use training transforms for train, eval transforms for val/test
        transform = TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM

        # Download and load the full BloodMNIST split
        dataset = BloodMNIST(
            split=split,
            transform=transform,
            download=True,
            root=data_dir,
        )

        # Extract indices for each hospital's class set
        healthy_indices = _extract_indices_by_class(dataset, HEALTHY_CLASSES)
        diseased_indices = _extract_indices_by_class(dataset, DISEASED_CLASSES)

        # Create PyTorch Subsets (zero-copy, references original dataset)
        partitions["hospital_a"][split] = Subset(dataset, healthy_indices)
        partitions["hospital_b"][split] = Subset(dataset, diseased_indices)

    return partitions


def save_partition_indices(
    data_dir: str = "./data",
) -> None:
    """
    Save partition indices to disk for reproducibility and Docker mounting.

    Saves two .pt files containing the index lists:
      - data/hospital_a_indices.pt
      - data/hospital_b_indices.pt

    Each file contains a dict: {"train": [...], "val": [...], "test": [...]}

    Args:
        data_dir: Root directory for dataset and index storage.
    """
    os.makedirs(data_dir, exist_ok=True)

    all_indices = {"hospital_a": {}, "hospital_b": {}}

    for split in ["train", "val", "test"]:
        dataset = BloodMNIST(
            split=split,
            transform=None,
            download=True,
            root=data_dir,
        )

        all_indices["hospital_a"][split] = _extract_indices_by_class(
            dataset, HEALTHY_CLASSES
        )
        all_indices["hospital_b"][split] = _extract_indices_by_class(
            dataset, DISEASED_CLASSES
        )

    # Persist to disk
    for hospital_id in ["hospital_a", "hospital_b"]:
        save_path = os.path.join(data_dir, f"{hospital_id}_indices.pt")
        torch.save(all_indices[hospital_id], save_path)
        print(f"  💾 Saved {hospital_id} indices → {save_path}")


def verify_partition(
    partitions: Dict[str, Dict[str, Subset]],
) -> bool:
    """
    Verify zero class overlap between hospital partitions.

    Asserts that Hospital A contains ONLY healthy classes and
    Hospital B contains ONLY diseased classes across all splits.

    Args:
        partitions: Output of partition_dataset().

    Returns:
        True if verification passes.

    Raises:
        AssertionError: If any class leaks between partitions.
    """
    print("\n🔍 Verifying partition integrity...")

    for split in ["train", "val", "test"]:
        # Check Hospital A — should only contain healthy classes
        for idx in partitions["hospital_a"][split].indices:
            dataset = partitions["hospital_a"][split].dataset
            label = int(dataset.labels[idx][0])
            assert label in HEALTHY_CLASSES, (
                f"❌ Hospital A [{split}] contains class {label} "
                f"({CLASS_NAMES[label]}) — expected only {HEALTHY_CLASSES}"
            )

        # Check Hospital B — should only contain diseased classes
        for idx in partitions["hospital_b"][split].indices:
            dataset = partitions["hospital_b"][split].dataset
            label = int(dataset.labels[idx][0])
            assert label in DISEASED_CLASSES, (
                f"❌ Hospital B [{split}] contains class {label} "
                f"({CLASS_NAMES[label]}) — expected only {DISEASED_CLASSES}"
            )

    print("  ✅ Zero class overlap confirmed across all splits!")
    return True


def print_partition_stats(
    partitions: Dict[str, Dict[str, Subset]],
) -> None:
    """
    Print detailed statistics for each hospital partition.

    Displays per-split sample counts and class distributions
    for visual inspection.

    Args:
        partitions: Output of partition_dataset().
    """
    print("\n" + "=" * 60)
    print("📊 NON-IID PARTITION STATISTICS")
    print("=" * 60)

    for hospital_id in ["hospital_a", "hospital_b"]:
        label = "🏥 Hospital A (Healthy)" if hospital_id == "hospital_a" \
                else "🏥 Hospital B (Diseased)"
        print(f"\n{label}")
        print("-" * 40)

        for split in ["train", "val", "test"]:
            subset = partitions[hospital_id][split]
            dataset = subset.dataset
            indices = subset.indices

            # Compute class distribution
            distribution = _compute_class_distribution(dataset, indices)

            print(f"  {split.upper():>5}: {len(indices):>6} samples")
            for class_name, count in distribution.items():
                print(f"         → {class_name}: {count}")

    # Total counts
    print(f"\n{'=' * 60}")
    total_a = sum(len(partitions["hospital_a"][s].indices) for s in ["train", "val", "test"])
    total_b = sum(len(partitions["hospital_b"][s].indices) for s in ["train", "val", "test"])
    print(f"  Total Hospital A: {total_a:>6} samples")
    print(f"  Total Hospital B: {total_b:>6} samples")
    print(f"  Grand Total:      {total_a + total_b:>6} samples")
    print("=" * 60)


def load_hospital_data(
    hospital_id: str,
    data_dir: str = "./data",
    split: str = "train",
) -> Subset:
    """
    Load a specific hospital's data partition for a given split.

    This is the primary entry point used by client_app.py to load
    the correct data partition based on the HOSPITAL_ID env var.

    Args:
        hospital_id: Either "hospital_a" or "hospital_b".
        data_dir: Root directory containing the dataset.
        split: One of "train", "val", or "test".

    Returns:
        A PyTorch Subset containing only this hospital's data.

    Raises:
        ValueError: If hospital_id is not recognized.
    """
    if hospital_id not in ("hospital_a", "hospital_b"):
        raise ValueError(
            f"Unknown hospital_id '{hospital_id}'. "
            f"Expected 'hospital_a' or 'hospital_b'."
        )

    # Determine class set for this hospital
    target_classes = HEALTHY_CLASSES if hospital_id == "hospital_a" else DISEASED_CLASSES

    # Select appropriate transform
    transform = TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM

    # Load the full dataset split
    dataset = BloodMNIST(
        split=split,
        transform=transform,
        download=True,
        root=data_dir,
    )

    # Filter to this hospital's classes
    indices = _extract_indices_by_class(dataset, target_classes)

    return Subset(dataset, indices)


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate extreme Non-IID BloodMNIST partitions for federated learning."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root directory for dataset download and storage (default: ./data)",
    )
    args = parser.parse_args()

    print("🚀 Starting Non-IID BloodMNIST Partitioning...")
    print(f"   Data directory: {os.path.abspath(args.data_dir)}")

    # Step 1: Download and partition
    partitions = partition_dataset(data_dir=args.data_dir)

    # Step 2: Print statistics
    print_partition_stats(partitions)

    # Step 3: Verify zero class overlap
    verify_partition(partitions)

    # Step 4: Save indices to disk
    print("\n💾 Saving partition indices...")
    save_partition_indices(data_dir=args.data_dir)

    print("\n✅ Partitioning complete! Ready for federated training.")
