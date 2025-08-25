"""
Package principal du projet de scoring de crédit
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"
__email__ = "data.science@company.com"

# Imports principaux
from .utils.config import get_feature_config, get_data_paths, get_model_paths
from .utils.logging import setup_logger, get_logger
from .data.schema import CreditDataValidator, CreditDataSchema
from .data.prepare import CreditDataPreparer

__all__ = [
    "get_feature_config",
    "get_data_paths", 
    "get_model_paths",
    "setup_logger",
    "get_logger",
    "CreditDataValidator",
    "CreditDataSchema",
    "CreditDataPreparer"
]



