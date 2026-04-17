"""
orchestrator.py
Governador de Computação do sistema AZR.

Responsabilidades:
  - Verificar se o sistema está ocioso antes de liberar experimentos.
  - Aplicar cooldowns em módulos que falharam no Watchdog/Sandbox.
  - Manter particionamento 80% produção / 20% sandbox via limites de CPU.
  - Registrar falhas persistentemente para que o AZR não repita experimentos ruins.
"""
import time
import json
import logging
import os
from typing import Optional

import psutil

log = logging.getLogger("AZR.Orchestrator")

# Limiares para decidir se o sistema está "ocioso"
_CPU_IDLE_THRESHOLD    = 40.0  # % CPU uso total abaixo disso = ocioso
_RAM_FREE_THRESHOLD_GB = 2.0   # RAM livre mínima para autorizar sandbox


class ComputeBudget:
    """Fatia de recursos que o sandbox pode usar."""
    sandbox_cpu_fraction: float = 0.20   # 20% dos núcleos
    production_cpu_fraction: float = 0.80

    @staticmethod
    def sandbox_max_cpus() -> float:
        logical = psutil.cpu_count(logical=True) or 1
        cpus = max(1.0, logical * ComputeBudget.sandbox_cpu_fraction)
        return round(cpus, 1)

    @staticmethod
    def sandbox_max_memory_mb() -> int:
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        # 20% da RAM total, mínimo de 512 MB
        return max(512, int(total_gb * 0.20 * 1024))


class Orchestrator:
    """
    Governador de Computação. Atua como porteiro para todos os experimentos
    do AZR, garantindo que o hardware principal nunca seja comprometido.
    """

    def __init__(self, failure_log_path: str = ".azr_failure_log.json"):
        self._cooldowns: dict[str, float] = {}         # component -> unix timestamp de expiração
        self._failure_log_path = failure_log_path
        self._failure_log: dict = self._load_failure_log()
        self.budget = ComputeBudget()

    # ------------------------------------------------------------------ #
    #  Failure log persistente                                             #
    # ------------------------------------------------------------------ #

    def _load_failure_log(self) -> dict:
        if os.path.exists(self._failure_log_path):
            try:
                with open(self._failure_log_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_failure_log(self):
        with open(self._failure_log_path, "w", encoding="utf-8") as f:
            json.dump(self._failure_log, f, indent=2, ensure_ascii=False)

    def register_failure(self, component: str, experiment_id: str, reason: str):
        """Registra falha no log persistente. O AZR lê este log para não repetir erros."""
        if component not in self._failure_log:
            self._failure_log[component] = []
        self._failure_log[component].append({
            "experiment_id": experiment_id,
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        })
        self._save_failure_log()
        log.warning(f"Falha registrada: [{component}] experimento='{experiment_id}' motivo='{reason}'")

    def get_failure_history(self, component: str) -> list:
        return self._failure_log.get(component, [])

    # ------------------------------------------------------------------ #
    #  Cooldown                                                            #
    # ------------------------------------------------------------------ #

    def add_cooldown(self, component: str, duration_sec: int = 21600):
        """
        Bloqueia um componente por duration_sec (padrão 6 horas = 21600s).
        """
        self._cooldowns[component] = time.time() + duration_sec
        log.info(f"Cooldown de {duration_sec}s aplicado ao componente '{component}'.")

    def _cooldown_remaining(self, component: str) -> Optional[float]:
        if component in self._cooldowns:
            remaining = self._cooldowns[component] - time.time()
            if remaining > 0:
                return remaining
            del self._cooldowns[component]
        return None

    # ------------------------------------------------------------------ #
    #  System idle check                                                   #
    # ------------------------------------------------------------------ #

    def _is_system_idle(self) -> bool:
        """
        Verifica se o sistema está suficientemente ocioso para liberar o sandbox.
        Regra: CPU abaixo do threshold E RAM livre suficiente.
        """
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        ram_free_gb = ram.available / (1024 ** 3)

        if cpu > _CPU_IDLE_THRESHOLD:
            log.debug(f"Sistema não ocioso: CPU={cpu:.1f}% (limiar={_CPU_IDLE_THRESHOLD}%)")
            return False
        if ram_free_gb < _RAM_FREE_THRESHOLD_GB:
            log.debug(f"Sistema não ocioso: RAM livre={ram_free_gb:.2f}GB (mínimo={_RAM_FREE_THRESHOLD_GB}GB)")
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Gate principal                                                      #
    # ------------------------------------------------------------------ #

    def can_run_experiment(self, component: str, require_idle: bool = True) -> tuple[bool, str]:
        """
        Verifica se é seguro rodar um experimento para este componente.

        Returns:
            (True, "") se liberado.
            (False, motivo) se bloqueado.
        """
        remaining = self._cooldown_remaining(component)
        if remaining is not None:
            mins = int(remaining / 60)
            return False, f"Componente '{component}' em cooldown por mais {mins} minutos."

        if require_idle and not self._is_system_idle():
            return False, "Sistema não ocioso. Aguardando baixo uso de CPU/RAM."

        return True, ""

    def get_sandbox_limits(self) -> dict:
        """Retorna os limites de hardware que o SandboxManager deve aplicar."""
        return {
            "max_cpus": self.budget.sandbox_max_cpus(),
            "max_memory_mb": self.budget.sandbox_max_memory_mb(),
        }
