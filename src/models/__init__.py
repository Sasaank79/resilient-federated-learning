"""
__init__.py — Models package initializer.

Exposes the model factory for convenient imports:
    from src.models import get_model
"""

from src.models.resnet import get_model, MedResNet18, count_parameters
