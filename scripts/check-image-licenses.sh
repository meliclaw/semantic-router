#!/usr/bin/env bash
# check-image-licenses.sh — fail on new/unapproved images in compose/Dockerfiles.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Allowlist: permissive or test-only. pinecone-local must never ship in prod images.
ALLOW_RE='^(postgres:17|pgvector/pgvector:pg17|qdrant/qdrant:|rust:|debian:bookworm-slim|python:[0-9.]+-slim|meliclaw-intent-router-service:)'
DENY_RE='(pinecone-local|element-web|minio/minio|postgres:latest)'

images=$(
  {
    find . \( -name 'docker-compose*.yml' -o -name 'docker-compose*.yaml' -o -name '*.compose.yaml' \) \
      -not -path './.git/*' -print0 2>/dev/null \
      | xargs -0 grep -hE '^\s*image:\s*' 2>/dev/null || true
    find . \( -name 'Dockerfile' -o -name 'Dockerfile.*' \) \
      -not -path './.git/*' -print0 2>/dev/null \
      | xargs -0 grep -hE '^FROM ' 2>/dev/null || true
  } | sed -E 's/.*image:[[:space:]]*//; s/^FROM[[:space:]]+//; s/ AS .*//; s/[[:space:]]*$//' | sort -u
)

failed=0
if echo "$images" | grep -E "$DENY_RE" >/dev/null 2>&1; then
  echo "BLOQUEANTE: imagen denegada (AGPL/tag flotante/test-only en prod):"
  echo "$images" | grep -E "$DENY_RE" || true
  failed=1
fi

# Flag floating postgres tag on FROM/image lines only (comments may mention it).
if grep -R -E '^(FROM|[[:space:]]*image:)[[:space:]].*postgres:latest' \
  --include='Dockerfile*' --include='*.yml' --include='*.yaml' . >/dev/null 2>&1; then
  echo "BLOQUEANTE: postgres:latest (tag flotante). Anclar pgvector/pgvector:pg17 o postgres:17."
  failed=1
fi

if [[ $failed -ne 0 ]]; then
  exit 1
fi
echo "Imágenes de compose/Dockerfile dentro de política (o ausentes en artefacto Rust)."
echo "Inventario detectado:"
echo "${images:-<ninguna>}"
