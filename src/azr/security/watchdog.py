"""
watchdog.py
Watchdog Independente do sistema AZR.

Responsabilidades:
  - Executar os Gabaritos de Ouro (arquivos .py com asserts nativos) após cada experimento.
  - Detectar Morte Cognitiva (falha de assert, timeout ou crash).
  - Coordenar: kill de container + rollback de snapshot + registro de falha no Orchestrator.
  - NÃO usa JSON, banco vetorial ou outra IA como árbitro — apenas exit code 0 vs não-zero.
"""
import subprocess
import logging
import os
from typing import Optional

log = logging.getLogger("AZR.Watchdog")


def _run_golden_test_file(test_file: str, timeout_sec: int) -> tuple[bool, str]:
    """
    Executa um único arquivo de gabarito Python e retorna (sucesso, log_detalhado).
    Alta performance: subprocesso nativo, sem overhead de framework de testes.
    """
    try:
        result = subprocess.run(
            ["python", test_file],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if result.returncode == 0:
            return True, ""
        detail = result.stderr.strip() or result.stdout.strip()
        return False, detail
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT após {timeout_sec}s"
    except Exception as e:
        return False, str(e)


class Watchdog:
    """
    Watchdog Independente.
    Integra-se ao SnapshotManager e Orchestrator para fechar o loop de failsafe.
    """

    def __init__(
        self,
        golden_tests_dir: str,
        snapshot_manager=None,
        orchestrator=None,
        test_timeout_sec: int = 30,
    ):
        """
        Args:
            golden_tests_dir:   Diretório com os arquivos .py de Gabarito de Ouro.
            snapshot_manager:   Instância de SnapshotManager para rollback.
            orchestrator:       Instância de Orchestrator para registrar falhas e cooldowns.
            test_timeout_sec:   Timeout por arquivo de teste.
        """
        self.golden_tests_dir = golden_tests_dir
        self.snapshot_manager = snapshot_manager
        self.orchestrator = orchestrator
        self.test_timeout_sec = test_timeout_sec

    # ------------------------------------------------------------------ #
    #  Execução dos testes                                                 #
    # ------------------------------------------------------------------ #

    def _collect_test_files(self) -> list[str]:
        """Coleta todos os arquivos .py no diretório de gabaritos."""
        if not os.path.isdir(self.golden_tests_dir):
            log.warning(f"Diretório de gabaritos não encontrado: '{self.golden_tests_dir}'")
            return []
        files = sorted(
            os.path.join(self.golden_tests_dir, f)
            for f in os.listdir(self.golden_tests_dir)
            if f.endswith(".py") and not f.startswith("_")
        )
        return files

    def run_all_golden_tests(self) -> tuple[bool, list[dict]]:
        """
        Executa todos os gabaritos em sequência.

        Returns:
            (all_passed: bool, results: list of {file, passed, detail})
        """
        test_files = self._collect_test_files()
        if not test_files:
            log.warning("Nenhum gabarito de ouro encontrado. Watchdog não pode validar.")
            return False, []

        results = []
        all_passed = True

        for tf in test_files:
            passed, detail = _run_golden_test_file(tf, self.test_timeout_sec)
            if passed:
                log.info(f"Gabarito OK: {os.path.basename(tf)}")
            else:
                log.error(f"Morte Cognitiva em '{os.path.basename(tf)}': {detail}")
                all_passed = False
            results.append({"file": tf, "passed": passed, "detail": detail})

        return all_passed, results

    # ------------------------------------------------------------------ #
    #  Container kill                                                      #
    # ------------------------------------------------------------------ #

    def kill_corrupted_container(self, container_name: str):
        """Destrói um container Docker corrompido ou fugitivo."""
        log.warning(f"Destruindo container corrompido: '{container_name}'")
        proc = subprocess.run(
            ["docker", "kill", container_name],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            log.info(f"Container '{container_name}' destruído com sucesso.")
        else:
            log.error(f"Falha ao destruir container '{container_name}': {proc.stderr.strip()}")

    # ------------------------------------------------------------------ #
    #  Loop de failsafe completo                                           #
    # ------------------------------------------------------------------ #

    def evaluate_experiment(
        self,
        component: str,
        experiment_id: str,
        container_name: Optional[str] = None,
        dest_code_dir: Optional[str] = None,
        dest_config: Optional[str] = None,
    ) -> bool:
        """
        Avalia um experimento recém-concluído:
          1. Roda todos os gabaritos de ouro.
          2. Se falhar: mata container, faz rollback, registra falha e aplica cooldown.
          3. Se passar: marca snapshot como 'watchdog-stable'.

        Returns:
            True se o experimento passou em todos os testes; False caso contrário.
        """
        log.info(f"Watchdog avaliando experimento '{experiment_id}' (componente: {component})...")
        all_passed, results = self.run_all_golden_tests()

        if all_passed:
            log.info(f"Experimento '{experiment_id}' APROVADO pelo Watchdog.")
            # Marca o snapshot mais recente como estável
            if self.snapshot_manager:
                snaps = self.snapshot_manager.list_snapshots()
                if snaps and snaps[0].get("experiment_id") == experiment_id:
                    # Atualiza o reason para 'watchdog-stable' no meta.json
                    import json, os as _os
                    meta_path = _os.path.join(
                        self.snapshot_manager.backup_dir,
                        snaps[0]["version_id"],
                        "meta.json",
                    )
                    snaps[0]["reason"] = "watchdog-stable"
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(snaps[0], f, indent=2)
            return True

        # ── FAILSAFE ──────────────────────────────────────────────────────
        failed = [r for r in results if not r["passed"]]
        reason = "; ".join(f"{os.path.basename(r['file'])}: {r['detail']}" for r in failed)

        # 1. Mata o container se ainda estiver rodando
        if container_name:
            self.kill_corrupted_container(container_name)

        # 2. Rollback
        if self.snapshot_manager and dest_code_dir and dest_config:
            stable_id = self.snapshot_manager.get_latest_stable_version()
            if stable_id:
                try:
                    self.snapshot_manager.restore_snapshot(stable_id, dest_code_dir, dest_config)
                    log.warning(f"ROLLBACK para versão estável '{stable_id}' concluído.")
                except Exception as e:
                    log.critical(f"Falha no rollback: {e}")
            else:
                log.critical("Nenhum snapshot estável disponível para rollback!")

        # 3. Registra falha e aplica cooldown
        if self.orchestrator:
            self.orchestrator.register_failure(component, experiment_id, reason)
            self.orchestrator.add_cooldown(component, duration_sec=21600)  # 6 horas

        return False
