"""
sandbox.py
Gerenciamento de containers Docker efêmeros para experimentos AZR.

Melhorias sobre a versão anterior:
  - Container recebe um nome único para poder ser 'docker kill'-ado pelo Watchdog.
  - Script é injetado via volume montado (tmpdir) e não passado como argumento
    (evita o problema de scripts > 4096 bytes no ARG do shell).
  - Suporte a GPU passthrough opcional (--gpus para experimentos que precisam de VRAM).
  - Retorno rico com container_id para integração com o Watchdog.
"""
import subprocess
import logging
import os
import uuid
import tempfile
import shutil

log = logging.getLogger("AZR.Sandbox")


class SandboxManager:
    """
    Gerenciamento de execução Docker. Instancia containers efêmeros, sem rede,
    com limites rígidos de CPU/Memória (cgroups), e com nome único para
    permitir kill externo pelo Watchdog.
    """

    def __init__(self, image_name: str = "azr_sandbox_base"):
        self.image_name = image_name

    def run_experiment(
        self,
        script_code: str,
        max_cpus: float = 1.0,
        max_memory_mb: int = 512,
        network_disabled: bool = True,
        allow_gpu: bool = False,
        timeout_sec: int = 120,
    ) -> dict:
        """
        Executa código Python dentro de um container Docker efêmero e isolado.

        Args:
            script_code:    Código Python a ser executado como string.
            max_cpus:       Limite de CPUs (float, ex: 1.5 = 1,5 núcleos).
            max_memory_mb:  Limite de RAM em MB.
            network_disabled: Se True, sem acesso à rede.
            allow_gpu:      Se True, passa --gpus=all (apenas para experimentos de performance).
            timeout_sec:    Tempo máximo de execução antes do kill externo.

        Returns:
            dict com keys: success, stdout, stderr, return_code, container_name, container_id
        """
        container_name = f"azr-exp-{uuid.uuid4().hex[:12]}"
        tmp_dir = tempfile.mkdtemp(prefix="azr_sandbox_")
        script_path = os.path.join(tmp_dir, "experiment.py")

        try:
            # Escreve o script em um diretório temporário que será montado no container
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_code)

            docker_cmd = [
                "docker", "run",
                "--rm",
                f"--name={container_name}",
                f"--cpus={max_cpus}",
                f"--memory={max_memory_mb}m",
                # Proteção adicional: sem chance de fork bomb
                "--pids-limit=256",
                # Monta o script como read-only
                f"--volume={tmp_dir}:/sandbox:ro",
            ]

            if network_disabled:
                docker_cmd.extend(["--network", "none"])

            if allow_gpu:
                docker_cmd.extend(["--gpus", "all"])

            # Imagem e comando final
            docker_cmd.extend([self.image_name, "/sandbox/experiment.py"])

            log.info(f"Sandbox '{container_name}': iniciando ({max_cpus} CPUs, {max_memory_mb}MB RAM)...")

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )

            success = result.returncode == 0
            if success:
                log.info(f"Sandbox '{container_name}': SUCESSO.")
            else:
                log.warning(
                    f"Sandbox '{container_name}': FALHOU (exit={result.returncode}).\n"
                    f"stderr: {result.stderr[:500]}"
                )

            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "container_name": container_name,
            }

        except subprocess.TimeoutExpired:
            log.error(
                f"Sandbox '{container_name}': TIMEOUT ({timeout_sec}s). "
                "Possível loop infinito. Destruindo container..."
            )
            # Tenta matar o container que pode ainda estar rodando
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            return {
                "success": False,
                "error": "timeout",
                "container_name": container_name,
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }
        finally:
            # Garante limpeza do diretório temporário
            shutil.rmtree(tmp_dir, ignore_errors=True)
