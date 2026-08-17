#!/usr/bin/env bash
# gen-third-party-notices.sh — regenerate crate license table via cargo-about.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HEADER_FILE="$(mktemp)"
trap 'rm -f "$HEADER_FILE"' EXIT

cat > "$HEADER_FILE" << 'EOF'
# Third-Party Notices

Meliclaw Intent Router incluye el siguiente software de terceros. Cada proyecto
permanece bajo su propia licencia; nada en este archivo altera esos términos.

Este archivo se regenera con `scripts/gen-third-party-notices.sh` (cargo-about
cuando está instalado). La entrada de origen del algoritmo se mantiene a mano
porque el port Rust es obra derivada, no un crate de crates.io.

---

## Origen del algoritmo (obra derivada)

**Proyecto:** semantic-router (biblioteca Python)
**Origen:** https://github.com/aurelio-labs/semantic-router
**Licencia:** MIT License
**Copyright:** Copyright (c) 2024 Aurelio AI
**NOTICE upstream:** ninguno
**Cambios de Meliclaw:** ver MELICLAW_PATCHES.md

Atribución factual de ingeniería. El nombre del proyecto upstream y de su
autor no se usan en material comercial ni como endoso.

**Crate Meliclaw:** `meliclaw-intent-router` (MIT, obra derivada)
**Pesos de modelos ONNX** (nomic-embed-text, bge-m3, etc.): licencia distinta
del código. Completar por despliegue.

---

## Crates de Rust (runtime)

EOF

if command -v cargo-about >/dev/null 2>&1; then
  cargo about generate about.hbs >> "$HEADER_FILE" 2>/dev/null \
    || cargo about generate --fail-on-missing > /tmp/about.md \
    && { echo "(tabla cargo-about)"; cat /tmp/about.md >> "$HEADER_FILE" || true; }
else
  cat >> "$HEADER_FILE" << 'EOF'
Regenerar la tabla con cargo-about instalado:

```bash
cargo install cargo-about
scripts/gen-third-party-notices.sh
```

Hasta entonces, `deny.toml` es la barrera bloqueante (AGPL, SSPL, BUSL/BSL,
Elastic, LGPL en artefacto).
EOF
fi

cp "$HEADER_FILE" "$ROOT/THIRD_PARTY_NOTICES.md"
echo "Wrote THIRD_PARTY_NOTICES.md"
