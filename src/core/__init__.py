"""
Módulo Core do SOREModel
Contém funcionalidades básicas como tokenização
"""

from .tokenizer import Tokenizer
from .tokenizer_pipeline import build_and_save_tokenizer

__all__ = [
    'Tokenizer',
    'build_and_save_tokenizer'
]
