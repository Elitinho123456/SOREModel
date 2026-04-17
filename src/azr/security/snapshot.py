import shutil
import os
import hashlib
import time

class SnapshotManager:
    """
    Gerencia o Snapshot Triplo: Código, Configuração e Pesos.
    Oferece métodos de hot-swap e rollback.
    """
    def __init__(self, backup_dir: str = ".azr_snapshots"):
        self.backup_dir = backup_dir
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def generate_hash(self, version_name: str) -> str:
        timestamp = str(time.time()).encode('utf-8')
        return f"{version_name}-{hashlib.md5(timestamp).hexdigest()[:8]}"

    def create_snapshot(self, code_dir: str, config_file: str, weights_dir: str = None) -> str:
        """Salva o estado atual e retorna a hash de versão."""
        version_hash = self.generate_hash("AZR-v1")
        target_dir = os.path.join(self.backup_dir, version_hash)
        os.makedirs(target_dir)

        # Copiar código
        if os.path.exists(code_dir):
            shutil.copytree(code_dir, os.path.join(target_dir, "code"))
            
        # Copiar config
        if os.path.exists(config_file):
            shutil.copy2(config_file, os.path.join(target_dir, "config.py"))

        # Copiar pesos (opcional se muito grande, preferível symlinks hard em prod)
        if weights_dir and os.path.exists(weights_dir):
             shutil.copytree(weights_dir, os.path.join(target_dir, "weights"))
             
        return version_hash

    def restore_snapshot(self, version_hash: str, dest_code_dir: str, dest_config: str):
        """No caso de Morte Cognitiva, reverte para a hash estável."""
        source_dir = os.path.join(self.backup_dir, version_hash)
        if not os.path.exists(source_dir):
            raise FileNotFoundError("Snapshot não encontrado!")

        if os.path.exists(dest_code_dir):
            shutil.rmtree(dest_code_dir)
        shutil.copytree(os.path.join(source_dir, "code"), dest_code_dir)

        shutil.copy2(os.path.join(source_dir, "config.py"), dest_config)
