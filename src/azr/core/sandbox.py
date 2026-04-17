import subprocess
import logging

class SandboxManager:
    """
    Gerenciamento de execução Docker. Instancia containers efêmeros isolados 
    de rede com limites rígidos de cgroups (hardware budget limits).
    """
    def __init__(self, image_name: str = "azr_sandbox_base"):
        self.image_name = image_name
        self.logger = logging.getLogger("AZR.Sandbox")

    def run_experiment(self, script_path: str, max_cpus: float = 1.0, max_memory_mb: int = 512, network_disabled: bool = True) -> dict:
        """
        Executa um código arbitrário dentro de um container Docker restrito e efêmero.
        """
        self.logger.info(f"Isolamento Absoluto: Iniciando sandbox para '{script_path}'...")
        
        # Parâmetros de proteção (cgroups limits)
        docker_cmd = [
            "docker", "run", "--rm",
            f"--cpus={max_cpus}",
            f"--memory={max_memory_mb}m"
        ]
        
        if network_disabled:
            docker_cmd.extend(["--network", "none"])
            
        docker_cmd.extend([
            # Em prod, montaria um volume ou passaria via stdin
            self.image_name,
            "python", script_path
        ])
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=120  # Watchdog de limite de tempo hard
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired as e:
            self.logger.error("Morte Cognitiva: O experimento estourou o tempo limite no Sandbox, sendo destruído.")
            return {
                "success": False,
                "error": "timeout",
                "details": str(e)
            }
