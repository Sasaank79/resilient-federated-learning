"""
resnet.py — Modified ResNet-18 for 28×28 Medical Image Classification

Architecture modifications from standard ResNet-18:
  1. First convolution: 3×3 (stride=1) instead of 7×7 (stride=2)
     — preserves spatial resolution for tiny 28×28 inputs
  2. Initial MaxPool removed — image too small for aggressive downsampling
  3. All BatchNorm layers replaced with GroupNorm (Opacus DP-SGD compatibility)
  4. Dropout (p=0.3) before final classifier for regularization
  5. Output head: 8 classes (BloodMNIST blood cell types)

Opacus Compatibility:
  Opacus's DP-SGD requires per-sample gradient isolation, which BatchNorm
  violates (it computes batch-wide statistics). GroupNorm operates within
  each sample independently, making it fully compatible.
  We use ModuleValidator.fix() as a safety net to catch any missed layers.

Usage:
    from src.models.resnet import get_model
    model = get_model(num_classes=8, dropout_rate=0.3)
"""

import torch
import torch.nn as nn
from torchvision import models
from opacus.validators import ModuleValidator

# ============================================================
# Default model hyperparameters
# ============================================================
DEFAULT_NUM_CLASSES = 8      # BloodMNIST has 8 blood cell types
DEFAULT_DROPOUT_RATE = 0.3   # Regularization dropout probability


class MedResNet18(nn.Module):
    """
    Modified ResNet-18 optimized for 28×28 medical image classification.

    This architecture adapts the standard ImageNet ResNet-18 for small
    medical images by removing aggressive downsampling and replacing
    BatchNorm with GroupNorm for differential privacy compatibility.

    Args:
        num_classes: Number of output classes (default: 8 for BloodMNIST).
        dropout_rate: Dropout probability before the final FC layer (default: 0.3).
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
    ):
        super().__init__()

        # --------------------------------------------------------
        # Load base ResNet-18 backbone (no pretrained weights)
        # --------------------------------------------------------
        backbone = models.resnet18(weights=None)

        # --------------------------------------------------------
        # Modification 1: Replace first conv for 28×28 input
        # Standard ResNet uses 7×7 conv with stride 2, which is
        # too aggressive for 28×28 images (would reduce to 14×14
        # immediately). We use 3×3 with stride 1 to preserve
        # spatial resolution.
        # --------------------------------------------------------
        backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # --------------------------------------------------------
        # Modification 2: Remove initial MaxPool
        # Standard ResNet applies MaxPool after first conv, halving
        # spatial dims again. At 28×28, this is too destructive.
        # Replace with identity (no-op).
        # --------------------------------------------------------
        backbone.maxpool = nn.Identity()

        # --------------------------------------------------------
        # Modification 3: Replace BatchNorm → GroupNorm
        # Opacus requires per-sample gradient computation, which
        # is incompatible with BatchNorm (batch-wide statistics).
        # ModuleValidator.fix() automatically replaces all
        # BatchNorm layers with GroupNorm equivalents.
        # --------------------------------------------------------
        backbone = ModuleValidator.fix(backbone)

        # --------------------------------------------------------
        # Modification 4: Replace classifier with Dropout + FC
        # Add dropout before the final fully-connected layer for
        # regularization, critical when training on small datasets.
        # --------------------------------------------------------
        in_features = backbone.fc.in_features  # 512 for ResNet-18
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the modified ResNet-18.

        Args:
            x: Input tensor of shape (batch_size, 3, 28, 28).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.backbone(x)


def get_model(
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
) -> MedResNet18:
    """
    Factory function to create a MedResNet18 model instance.

    Creates the model and validates Opacus compatibility as a safety check.

    Args:
        num_classes: Number of output classes (default: 8).
        dropout_rate: Dropout probability (default: 0.3).

    Returns:
        A validated MedResNet18 model ready for DP-SGD training.
    """
    model = MedResNet18(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )

    # --------------------------------------------------------
    # Safety validation: ensure all layers are Opacus-compatible
    # This should pass since we used ModuleValidator.fix() above,
    # but we check explicitly as a production safeguard.
    # --------------------------------------------------------
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"⚠️  Opacus validation found {len(errors)} issue(s), attempting auto-fix...")
        model = ModuleValidator.fix(model)

        # Re-validate after fix
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            raise RuntimeError(
                f"❌ Model is still incompatible with Opacus after fix: {errors}"
            )

    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in the model.

    Args:
        model: A PyTorch module.

    Returns:
        Dictionary with 'total', 'trainable', and 'non_trainable' counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


# ============================================================
# CLI — Quick model inspection
# ============================================================
if __name__ == "__main__":
    print("🏗️  Building MedResNet18 for BloodMNIST (28×28×3, 8 classes)...")

    model = get_model()
    params = count_parameters(model)

    print(f"\n📐 Architecture Summary:")
    print(f"   Total parameters:       {params['total']:>10,}")
    print(f"   Trainable parameters:   {params['trainable']:>10,}")
    print(f"   Non-trainable params:   {params['non_trainable']:>10,}")

    # Validate with a dummy forward pass
    dummy_input = torch.randn(4, 3, 28, 28)
    output = model(dummy_input)
    print(f"\n🧪 Forward pass test:")
    print(f"   Input shape:  {tuple(dummy_input.shape)}")
    print(f"   Output shape: {tuple(output.shape)}")
    print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Opacus compatibility check
    errors = ModuleValidator.validate(model, strict=False)
    status = "✅ PASSED" if not errors else f"❌ FAILED ({len(errors)} errors)"
    print(f"\n🔒 Opacus DP-SGD Compatibility: {status}")

    print("\n✅ Model ready for federated differential-privacy training!")
