# Meliclaw Intent Router (fork)

Producto interno: **Meliclaw Intent Classifier**.

- Crate: [`meliclaw-intent-router`](crates/meliclaw-intent-router)
- Rama de trabajo: `meliclaw-intent-router-main`
- Espejo upstream: `main` (no modificar)
- HTTP facade y hybrid/indexes: PRs siguientes

Port Rust (MIT, obra derivada) del algoritmo publicado en
https://github.com/aurelio-labs/semantic-router
(Copyright 2024 Aurelio AI, MIT). Texto de licencia en [`LICENSE`](LICENSE).
Atribución factual: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
Parches: [`MELICLAW_PATCHES.md`](MELICLAW_PATCHES.md).

El árbol Python del upstream permanece como referencia. El artefacto de
Meliclaw es el workspace Cargo.

```bash
cargo test -p meliclaw-intent-router
```

Capas 2–3 (waterfall + RRF) **no** están en este repositorio.
