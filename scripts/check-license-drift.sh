#!/usr/bin/env bash
# check-license-drift.sh — fails if LICENSE / NOTICE drifted vs upstream.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! git remote get-url upstream >/dev/null 2>&1; then
  echo "BLOQUEANTE: remote 'upstream' no configurado."
  exit 1
fi

git fetch upstream --quiet || true
UPSTREAM_REF="${UPSTREAM_REF:-upstream/main}"

failed=0
for f in LICENSE LICENSE-MIT LICENSE-APACHE NOTICE; do
  if git diff --name-only "${UPSTREAM_REF}...HEAD" -- "$f" 2>/dev/null | grep -q .; then
    echo "BLOQUEANTE: '$f' cambió respecto a ${UPSTREAM_REF}. Revisión legal requerida."
    failed=1
  fi
done

if [[ -f LICENSE ]]; then
  if ! grep -q "Copyright (c) 2024 Aurelio AI" LICENSE; then
    echo "BLOQUEANTE: LICENSE no conserva el copyright de Aurelio AI."
    failed=1
  fi
  if ! grep -q "MIT License" LICENSE; then
    echo "BLOQUEANTE: LICENSE no es MIT."
    failed=1
  fi
else
  echo "BLOQUEANTE: falta LICENSE."
  failed=1
fi

if [[ $failed -ne 0 ]]; then
  exit 1
fi
echo "Sin deriva en archivos de licencia."
