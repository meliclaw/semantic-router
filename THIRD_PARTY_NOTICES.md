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

Regenerar la tabla con:

```bash
scripts/gen-third-party-notices.sh
```

Hasta que cargo-about esté instalado en CI, el deny-list de `deny.toml` es la
barrera bloqueante (AGPL, SSPL, BUSL/BSL, Elastic, LGPL en artefacto).
