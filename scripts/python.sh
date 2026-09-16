#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$HUMANOID_BADMINTON_ROOT/.venv/bin/python" "$@"
