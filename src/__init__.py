"""
SOREModel - Simple Open-Source Recurrent/Transformer Model
Módulo principal do projeto SOREModel
"""

from .core import Tokenizer
from .models import SOREModel_v2, atentionHead, MultiHeadAttention, Block
from .training import Trainer
from .generation import TextGenerator

__version__ = "2.0.0"
__author__ = "SOREModel Team"
__description__ = "Simple Open-Source Recurrent/Transformer Model for text generation"

__all__ = [
    # Core
    'Tokenizer',
    # Models
    'SOREModel_v2',
    'atentionHead',
    'MultiHeadAttention',
    'Block',
    # Training
    'Trainer',
    # Generation
    'TextGenerator'
]
