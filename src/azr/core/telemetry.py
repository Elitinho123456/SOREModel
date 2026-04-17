import psutil
import subprocess
import json
import logging

class Telemetry:
    """
    Percepção de Hardware: Monitora CPU, RAM e uso de GPU (nvidia-smi) 
    para alimentar o prompt do AZR com contexto de hardware.
    """
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AZR.Telemetry")

    def get_gpu_metrics_nvidia_smi(self):
        """Retorna as métricas da NVIDIA de forma performática via nvidia-smi."""
        try:
            # Puxando as métricas diretamente no formato CSV
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 5:
                        gpus.append({
                            "index": int(parts[0]),
                            "utilization_gpu_percent": float(parts[1]),
                            "memory_used_mb": float(parts[2]),
                            "memory_total_mb": float(parts[3]),
                            "temperature_celsius": float(parts[4])
                        })
            return gpus
        except FileNotFoundError:
            self.logger.warning("nvidia-smi não encontrado no PATH. Usando mock.")
            return [{"error": "nvidia-smi missing", "mock": True}]
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Erro ao executar nvidia-smi: {e}")
            return [{"error": "execution failed", "details": str(e)}]

    def get_hardware_status(self) -> dict:
        """Dashboard de Sensores."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')

        status = {
            "system": {
                "cpu_utilization_percent": cpu_percent,
                "ram_used_gb": round(ram_info.used / (1024**3), 2),
                "ram_total_gb": round(ram_info.total / (1024**3), 2),
                "disk_usage_percent": disk_info.percent
            },
            "gpus": self.get_gpu_metrics_nvidia_smi()
        }
        
        return status

if __name__ == "__main__":
    t = Telemetry()
    print(json.dumps(t.get_hardware_status(), indent=2))
