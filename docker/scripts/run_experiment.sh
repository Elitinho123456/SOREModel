#!/bin/bash
# Script utilitário para envolver o docker run do lado de fora do sistema
# e ser invocado pelo orquestrador do AZR.

MAX_CPUS="1.0"
MAX_MEM="512m"
CMD_TO_RUN=$1

echo "[AZR] Montando Sandbox para: $CMD_TO_RUN"

docker run --rm \
    --cpus "$MAX_CPUS" \
    --memory "$MAX_MEM" \
    --network none \
    azr_sandbox_base \
    -c "$CMD_TO_RUN"
