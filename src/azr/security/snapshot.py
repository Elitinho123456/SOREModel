"""
snapshot.py
Gerenciador de Snapshot Triplo do sistema AZR.

O Snapshot Triplo é composto por:
  1. Código-fonte do executor (src/azr)
  2. Arquivo de configuração (config.py)
  3. Checkpoints/Pesos do modelo (opcional — pode ser muito grande)

Cada snapshot recebe uma hash de versão baseada em SHA-256 do conteúdo, não 
apenas do timestamp (resistente a colisões em máquinas rápidas).

Adiciona suporte a:
  - Listagem e inspeção de snapshots salvos.
  - Registro de metadados (motivo, timestamp, experimento que originou).
  - Limpeza automática de snapshots antigos (mantém os N mais recentes).
"""
import shutil
import os
import hashlib
import time
import json
import logging
from typing import Optional

log = logging.getLogger("AZR.Snapshot")


class SnapshotManager:
    """
    Gerencia o Snapshot Triplo: Código, Configuração e Pesos.
    Oferece métodos para criação, restauração e limpeza de snapshots.
    """

    def __init__(self, backup_dir: str = ".azr_snapshots", max_keep: int = 10):
        """
        Args:
            backup_dir: Diretório raiz onde os snapshots serão salvos.
            max_keep:   Número máximo de snapshots a manter. Os mais antigos são purgados.
        """
        self.backup_dir = backup_dir
        self.max_keep = max_keep
        os.makedirs(self.backup_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Hash de versão                                                      #
    # ------------------------------------------------------------------ #

    def _compute_dir_hash(self, path: str) -> str:
        """Gera SHA-256 do conteúdo de um diretório (arquivos .py ordenados)."""
        sha = hashlib.sha256()
        for root, _, files in os.walk(path):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "rb") as f:
                        sha.update(f.read())
                except OSError:
                    pass
        return sha.hexdigest()[:16]

    def _generate_version_id(self, code_dir: str) -> str:
        content_hash = self._compute_dir_hash(code_dir)
        ts = time.strftime("%Y%m%d-%H%M%S")
        return f"AZR-{ts}-{content_hash}"

    # ------------------------------------------------------------------ #
    #  Criação de snapshot                                                 #
    # ------------------------------------------------------------------ #

    def create_snapshot(
        self,
        code_dir: str,
        config_file: str,
        weights_dir: Optional[str] = None,
        reason: str = "manual",
        experiment_id: Optional[str] = None,
    ) -> str:
        """
        Salva o estado atual (Snapshot Triplo) e retorna o version_id.

        Args:
            code_dir:       Diretório com o código-fonte (ex: src/azr).
            config_file:    Arquivo de configuração (ex: config.py).
            weights_dir:    Diretório com checkpoints do modelo (opcional).
            reason:         Motivo do snapshot (ex: 'pre-experiment', 'watchdog-stable').
            experiment_id:  ID do experimento que originou este snapshot (para rastreabilidade).
        """
        version_id = self._generate_version_id(code_dir)
        target_dir = os.path.join(self.backup_dir, version_id)

        if os.path.exists(target_dir):
            log.warning(f"Snapshot '{version_id}' já existe. Sobrescrevendo.")
            shutil.rmtree(target_dir)

        os.makedirs(target_dir)
        log.info(f"Criando snapshot '{version_id}'...")

        # 1. Código
        if os.path.exists(code_dir):
            shutil.copytree(code_dir, os.path.join(target_dir, "code"))
        else:
            log.warning(f"code_dir '{code_dir}' não encontrado. Snapshot parcial.")

        # 2. Config
        if os.path.exists(config_file):
            shutil.copy2(config_file, os.path.join(target_dir, "config.py"))
        else:
            log.warning(f"config_file '{config_file}' não encontrado. Snapshot parcial.")

        # 3. Pesos (opcional — pode ser muito grande)
        if weights_dir and os.path.exists(weights_dir):
            shutil.copytree(weights_dir, os.path.join(target_dir, "weights"))

        # 4. Metadados do snapshot
        meta = {
            "version_id": version_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reason": reason,
            "experiment_id": experiment_id,
            "has_weights": weights_dir is not None and os.path.exists(weights_dir or ""),
        }
        with open(os.path.join(target_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        log.info(f"Snapshot '{version_id}' salvo com sucesso.")
        self._purge_old_snapshots()
        return version_id

    # ------------------------------------------------------------------ #
    #  Restauração                                                         #
    # ------------------------------------------------------------------ #

    def restore_snapshot(
        self,
        version_id: str,
        dest_code_dir: str,
        dest_config: str,
        restore_weights_to: Optional[str] = None,
    ):
        """
        Reverte para o estado de um snapshot (Morte Cognitiva / Failsafe).

        Args:
            version_id:         ID do snapshot a restaurar.
            dest_code_dir:      Onde restaurar o código.
            dest_config:        Caminho onde restaurar o config.py.
            restore_weights_to: Caminho onde restaurar os pesos (opcional).
        """
        source_dir = os.path.join(self.backup_dir, version_id)
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Snapshot '{version_id}' não encontrado em '{self.backup_dir}'.")

        log.warning(f"ROLLBACK: Restaurando snapshot '{version_id}'...")

        # Código
        src_code = os.path.join(source_dir, "code")
        if os.path.exists(src_code):
            if os.path.exists(dest_code_dir):
                shutil.rmtree(dest_code_dir)
            shutil.copytree(src_code, dest_code_dir)
            log.info(f"Código restaurado em '{dest_code_dir}'.")

        # Config
        src_cfg = os.path.join(source_dir, "config.py")
        if os.path.exists(src_cfg):
            shutil.copy2(src_cfg, dest_config)
            log.info(f"Config restaurado em '{dest_config}'.")

        # Pesos (opcional)
        src_weights = os.path.join(source_dir, "weights")
        if restore_weights_to and os.path.exists(src_weights):
            if os.path.exists(restore_weights_to):
                shutil.rmtree(restore_weights_to)
            shutil.copytree(src_weights, restore_weights_to)
            log.info(f"Pesos restaurados em '{restore_weights_to}'.")

        log.warning(f"ROLLBACK CONCLUÍDO para '{version_id}'.")

    # ------------------------------------------------------------------ #
    #  Listagem e limpeza                                                  #
    # ------------------------------------------------------------------ #

    def list_snapshots(self) -> list[dict]:
        """Lista todos os snapshots disponíveis (mais recente primeiro)."""
        snapshots = []
        for name in os.listdir(self.backup_dir):
            meta_path = os.path.join(self.backup_dir, name, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    snapshots.append(json.load(f))
        return sorted(snapshots, key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_latest_stable_version(self) -> Optional[str]:
        """Retorna o version_id do snapshot mais recente com reason='watchdog-stable'."""
        for snap in self.list_snapshots():
            if snap.get("reason") == "watchdog-stable":
                return snap["version_id"]
        # Fallback: retorna o mais recente disponível
        snaps = self.list_snapshots()
        return snaps[0]["version_id"] if snaps else None

    def _purge_old_snapshots(self):
        """Remove snapshots mais antigos além do limite max_keep."""
        snapshots = self.list_snapshots()
        to_delete = snapshots[self.max_keep:]
        for snap in to_delete:
            vid = snap["version_id"]
            path = os.path.join(self.backup_dir, vid)
            shutil.rmtree(path, ignore_errors=True)
            log.info(f"Snapshot antigo purgado: '{vid}'.")
