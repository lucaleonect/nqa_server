#!/usr/bin/env bash
set -euo pipefail
# Output JSON describing available GPUs; Spark expects: {"name":"gpu","addresses":["<id>", ...]}
mapfile -t GPUS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
printf '{"name":"gpu","addresses":[%s]}' "$(printf '"%s",' "${GPUS[@]}" | sed 's/,$//')"