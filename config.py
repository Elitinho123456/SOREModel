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
    'dim_embed': 768,        # Updated to v3/v4 default
    'tamanho_contexto': 1024,
    'num_heads': 12,
    'num_layers': 12,
    'learning_rate': 6e-4,
    'batch_size': 8,
    'num_epocas': 10
}

# Configurações de Treino Avançado
TRAINING_CONFIG = {
    'use_amp': True,
    'lr_scheduler': "cosine", # options: "none", "step", "cosine"
    'warmup_steps': 500,
    'early_stopping_patience': 3,
    'min_delta': 0.0,
    'num_workers': 4,       # Adjust based on CPU
    'pin_memory': True,
    'prefetch_factor': 2
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
