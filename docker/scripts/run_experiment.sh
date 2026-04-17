#!/bin/bash
# run_experiment.sh
# Envolve o docker run do lado de fora, integrado ao Orchestrator do AZR.
# Invocado como: ./docker/scripts/run_experiment.sh <script.py> [max_cpus] [max_mem_mb]
#
# Uso:
#   ./docker/scripts/run_experiment.sh /tmp/azr_sandbox_xyz/experiment.py 1.5 768
#
# Saída:
#   Exit code 0 = sucesso; Exit code não-zero = falha (Watchdog intercepta).

set -euo pipefail

SCRIPT_PATH="${1:?Informe o caminho do script Python}"
MAX_CPUS="${2:-1.0}"
MAX_MEM_MB="${3:-512}"
CONTAINER_NAME="azr-exp-$(cat /proc/sys/kernel/random/uuid 2>/dev/null | head -c 8 || date +%s)"

echo "[AZR] Montando Sandbox: container='${CONTAINER_NAME}' cpus=${MAX_CPUS} mem=${MAX_MEM_MB}m"
echo "[AZR] Script: ${SCRIPT_PATH}"

# Monta o diretório pai do script como volume read-only
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
SCRIPT_FILE="$(basename "${SCRIPT_PATH}")"

docker run \
    --rm \
    --name "${CONTAINER_NAME}" \
    --cpus "${MAX_CPUS}" \
    --memory "${MAX_MEM_MB}m" \
    --pids-limit 256 \
    --network none \
    --volume "${SCRIPT_DIR}:/sandbox:ro" \
    azr_sandbox_base \
    "/sandbox/${SCRIPT_FILE}"

EXIT_CODE=$?

if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "[AZR] Sandbox SUCESSO: '${CONTAINER_NAME}'"
else
    echo "[AZR] Sandbox FALHOU (exit=${EXIT_CODE}): '${CONTAINER_NAME}'"
fi

exit "${EXIT_CODE}"
