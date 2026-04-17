"""
telemetry.py
Dashboard de Sensores do AZR.

Puxa em alta velocidade via nvidia-smi (CSV, zero overhead de Python bindings)
e via psutil para CPU/RAM/Disk. Implementa cache de intervalo para evitar
múltiplas chamadas em cadência muito alta.
"""
import subprocess
import json
import logging
import time
from typing import Optional

import psutil

log = logging.getLogger("AZR.Telemetry")

_NVIDIA_SMI_FIELDS = (
    "index,"
    "utilization.gpu,"
    "memory.used,"
    "memory.total,"
    "temperature.gpu,"
    "power.draw,"
    "clocks.current.sm"
)


class Telemetry:
    """
    Percepção de Hardware: Monitora CPU, RAM, GPU (via nvidia-smi), Disco e I/O
    para alimentar o prompt de contexto do AZR com dados de hardware em tempo real.
    """

    def __init__(self, cache_ttl_sec: float = 1.0):
        """
        Args:
            cache_ttl_sec: Intervalo mínimo entre queries reais ao nvidia-smi.
                           Chamadas mais frequentes retornam o cache anterior.
        """
        self._cache_ttl = cache_ttl_sec
        self._gpu_cache: Optional[list] = None
        self._gpu_cache_ts: float = 0.0

    # ------------------------------------------------------------------ #
    #  GPU (nvidia-smi)                                                    #
    # ------------------------------------------------------------------ #

    def _query_nvidia_smi(self) -> list:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={_NVIDIA_SMI_FIELDS}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,  # nvidia-smi nunca deveria demorar mais que isso
            )

            gpus = []
            for line in result.stdout.strip().splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue

                def _safe_float(val: str, fallback: float = -1.0) -> float:
                    try:
                        return float(val)
                    except ValueError:
                        return fallback

                gpu = {
                    "index": int(parts[0]),
                    "utilization_gpu_percent": _safe_float(parts[1]),
                    "memory_used_mb": _safe_float(parts[2]),
                    "memory_total_mb": _safe_float(parts[3]),
                    "temperature_celsius": _safe_float(parts[4]),
                }
                if len(parts) > 5:
                    gpu["power_draw_w"] = _safe_float(parts[5])
                if len(parts) > 6:
                    gpu["sm_clock_mhz"] = _safe_float(parts[6])

                # Derivadas úteis para o AZR
                if gpu["memory_total_mb"] > 0:
                    gpu["memory_free_mb"] = gpu["memory_total_mb"] - gpu["memory_used_mb"]
                    gpu["memory_used_percent"] = round(
                        100.0 * gpu["memory_used_mb"] / gpu["memory_total_mb"], 1
                    )

                gpus.append(gpu)
            return gpus

        except FileNotFoundError:
            log.warning("nvidia-smi não encontrado no PATH.")
            return [{"available": False, "reason": "nvidia-smi missing"}]
        except subprocess.TimeoutExpired:
            log.warning("nvidia-smi timeout — GPU pode estar travada.")
            return [{"available": False, "reason": "nvidia-smi timeout"}]
        except subprocess.CalledProcessError as e:
            log.error(f"nvidia-smi retornou erro: {e.stderr.strip()}")
            return [{"available": False, "reason": str(e)}]

    def get_gpu_metrics(self) -> list:
        """Retorna métricas da GPU com cache TTL para não saturar o barramento."""
        now = time.monotonic()
        if self._gpu_cache is None or (now - self._gpu_cache_ts) >= self._cache_ttl:
            self._gpu_cache = self._query_nvidia_smi()
            self._gpu_cache_ts = now
        return self._gpu_cache

    # ------------------------------------------------------------------ #
    #  Sistema                                                             #
    # ------------------------------------------------------------------ #

    def get_cpu_metrics(self) -> dict:
        """CPU por núcleo e média geral."""
        per_core = psutil.cpu_percent(percpu=True, interval=0.1)
        return {
            "total_percent": round(sum(per_core) / len(per_core), 1),
            "per_core_percent": per_core,
            "num_cores": len(per_core),
        }

    def get_ram_metrics(self) -> dict:
        v = psutil.virtual_memory()
        return {
            "used_gb": round(v.used / 1024 ** 3, 2),
            "available_gb": round(v.available / 1024 ** 3, 2),
            "total_gb": round(v.total / 1024 ** 3, 2),
            "used_percent": v.percent,
        }

    def get_disk_metrics(self, path: str = "/") -> dict:
        """Uso de disco e I/O counters."""
        usage = psutil.disk_usage(path)
        io = psutil.disk_io_counters()
        return {
            "path": path,
            "used_gb": round(usage.used / 1024 ** 3, 2),
            "free_gb": round(usage.free / 1024 ** 3, 2),
            "total_gb": round(usage.total / 1024 ** 3, 2),
            "used_percent": usage.percent,
            "read_mb_total": round(io.read_bytes / 1024 ** 2, 1) if io else -1,
            "write_mb_total": round(io.write_bytes / 1024 ** 2, 1) if io else -1,
        }

    # ------------------------------------------------------------------ #
    #  Dashboard unificado (para injeção no system prompt do AZR)         #
    # ------------------------------------------------------------------ #

    def get_hardware_status(self) -> dict:
        """
        Retorna snapshot completo do hardware.
        Este dicionário é serializado em JSON e prefixado no system prompt do AZR.
        """
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cpu": self.get_cpu_metrics(),
            "ram": self.get_ram_metrics(),
            "disk": self.get_disk_metrics(),
            "gpus": self.get_gpu_metrics(),
        }

    def format_for_prompt(self) -> str:
        """Formata o status em texto compacto para injeção no system prompt."""
        s = self.get_hardware_status()
        gpu_lines = []
        for g in s["gpus"]:
            if not g.get("available", True) is False:
                gpu_lines.append(
                    f"  GPU[{g.get('index', '?')}]: "
                    f"util={g.get('utilization_gpu_percent', '?')}% "
                    f"vram={g.get('memory_used_mb', '?')}/{g.get('memory_total_mb', '?')}MB "
                    f"temp={g.get('temperature_celsius', '?')}°C"
                )
            else:
                gpu_lines.append(f"  GPU: {g.get('reason', 'unavailable')}")

        lines = [
            f"[AZR TELEMETRY {s['timestamp']}]",
            f"CPU: {s['cpu']['total_percent']}%  RAM: {s['ram']['used_gb']}/{s['ram']['total_gb']}GB ({s['ram']['used_percent']}%)",
            f"Disk({s['disk']['path']}): {s['disk']['used_percent']}% used  free={s['disk']['free_gb']}GB",
        ] + gpu_lines
        return "\n".join(lines)


if __name__ == "__main__":
    t = Telemetry()
    print(json.dumps(t.get_hardware_status(), indent=2))
    print()
    print(t.format_for_prompt())
