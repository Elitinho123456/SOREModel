"""
Módulo de Modelos do SOREModel
Contém as diferentes versões e arquiteturas do modelo
"""

from .soreModel_v2 import (
    atentionHead,
    MultiHeadAttention,
    Block,
    SOREModel_v2
)

__all__ = [
    'atentionHead',
    'MultiHeadAttention',
    'Block',
    'SOREModel_v2'
]
