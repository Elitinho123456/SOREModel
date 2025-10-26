"""
Configurações do projeto SOREModel
Arquivo para facilitar desenvolvimento e uso
"""

import os

# Configurações de desenvolvimento
DEBUG = True
LOG_LEVEL = "INFO"

# Configurações do modelo padrão
DEFAULT_MODEL_CONFIG = {
    'dim_embed': 64,
    'tamanho_contexto': 16,
    'num_heads': 8,
    'num_layers': 4,
    'learning_rate': 0.001,
    'batch_size': 4,
    'num_epocas': 1000
}

# Configurações de geração de texto
DEFAULT_GENERATION_CONFIG = {
    'max_length': 50,
    'temperature': 0.8,
    'top_k': 40,
    'top_p': 0.9,
    'beam_width': 3
}

# Configurações de paths
PATHS = {
    'src': os.path.join(os.path.dirname(__file__), 'src'),
    'tests': os.path.join(os.path.dirname(__file__), 'tests'),
    'examples': os.path.join(os.path.dirname(__file__), 'examples'),
    'docs': os.path.join(os.path.dirname(__file__), 'docs'),
    'scripts': os.path.join(os.path.dirname(__file__), 'scripts'),
    'data': os.path.join(os.path.dirname(__file__), 'data')
}

# Adicionar src ao path do Python automaticamente
import sys
if PATHS['src'] not in sys.path:
    sys.path.append(PATHS['src'])

# Versionamento
__version__ = "2.0.0"
__author__ = "SOREModel Team"
__description__ = "Simple Open-Source Recurrent/Transformer Model"
