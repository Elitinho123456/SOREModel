import subprocess
import logging
import ast

class Watchdog:
    """
    Independente e inflexível.
    Verifica a performance da nova implementação executando 'Gabaritos de Ouro' em arquivos python independentes puramente com Asserts.
    Se falhar, dispara o aviso de rollback.
    """
    def __init__(self, golden_test_file: str):
        self.golden_test_file = golden_test_file
        self.logger = logging.getLogger("AZR.Watchdog")

    def run_golden_test(self, timeout_sec: int = 30) -> bool:
        """Roda um arquivo Python e espera exit code 0. Alta performance por não parsear JSON/DB."""
        self.logger.info(f"Executando Teste Gabarito: {self.golden_test_file}")
        try:
            result = subprocess.run(
                ["python", self.golden_test_file],
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            if result.returncode == 0:
                self.logger.info("Teste de Ouro: SUCESSO. Funcionalidade intacta.")
                return True
            else:
                self.logger.error(f"Morte Cognitiva (Assert Error). Log: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error("Morte Cognitiva (Timeout). Possível loop infinito ou gargalo inserido.")
            return False
            
    def kill_corrupted_container(self, container_id: str):
        """Método de segurança para intervir e destruir um sandbox fugitivo via shell."""
        subprocess.run(["docker", "kill", container_id])
        self.logger.warning(f"Container corrompido {container_id} destruído pelo Watchdog.")
