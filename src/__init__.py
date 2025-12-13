"""
SOREModel - Simple Open-Source Recurrent/Transformer Model
"""

from .core import Tokenizer
from .models import SOREModel_v3
from .training import Trainer
from .data.dataset import TextDataset
from .generation import TextGenerator

__version__ = "2.1.0"
__author__ = "SOREModel Team"

__all__ = [
    'Tokenizer',
    'SOREModel_v3',
    'Trainer',
    'TextDataset',
    'TextGenerator'
]
