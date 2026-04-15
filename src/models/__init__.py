from .soreModel_v3 import SOREModel_v3, ModelConfig as ModelConfigV3
from .soreModel_v4_1 import SOREModel_v4_1, ModelConfig as ModelConfigV4_1

# Defaults to v4.1 while preserving explicit v3 imports.
SOREModel = SOREModel_v4_1
ModelConfig = ModelConfigV4_1

__all__ = [
	'SOREModel',
	'ModelConfig',
	'SOREModel_v3',
	'ModelConfigV3',
	'SOREModel_v4_1',
	'ModelConfigV4_1',
]
