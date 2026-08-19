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

---

## Pesos de modelos de embedding (no son el código MIT)

Las licencias de **pesos** son independientes del crate. No se vendorizan en
este repo. Completar la línea de atribución por despliegue.

| Modelo | Id canónico | Dim | Licencia de pesos | Notas |
|---|---|---|---|---|
| nomic-embed-text-v1.5 | `nomic-ai/nomic-embed-text-v1.5` | 768 | ver tarjeta del modelo | Default CPU/edge ONNX |
| bge-m3 | `BAAI/bge-m3` | 1024 | ver tarjeta del modelo | ONNX |
| **Nemotron-3-Embed-1B-BF16** | `nvidia/Nemotron-3-Embed-1B-BF16` | **2048** | **OpenMDW-1.1** (NVIDIA; uso comercial permitido; Linux Foundation). Built with Ministral-3-3B-Instruct-2512 (Apache-2.0). | Capa 1 GPU / HTTP. Tag Ollama/NIM: `nemotron-3-embed-1b`. No hay URL ONNX oficial de NVIDIA. Export comunitario opcional: `kzzalews/Nemotron-3-Embed-1B-BF16-onnx` (`model.onnx`, `fp16/model.onnx`). Verificar checksum contra pesos NVIDIA antes de producción. |

**ColNomic Embed Multimodal 7B** no se cablea en este crate (retrieval visual /
late-interaction, no un `Vec<f32>` para cosine de Capa 1).

Texto OpenMDW-1.1: tarjeta del modelo en
https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16
y https://www.linuxfoundation.org/legal/openmdw-license

---

## Crates de Rust (runtime)

Regenerar la tabla con:

```bash
scripts/gen-third-party-notices.sh
```

Hasta que cargo-about esté instalado en CI, el deny-list de `deny.toml` es la
barrera bloqueante (AGPL, SSPL, BUSL/BSL, Elastic, LGPL en artefacto).
