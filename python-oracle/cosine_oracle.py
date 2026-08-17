#!/usr/bin/env python3
"""Golden cosine vs numpy linear.py. Run from repo root with numpy installed.

    uv run python python-oracle/cosine_oracle.py

Prints JSON used by crates/meliclaw-intent-router tests (values are also
hard-coded so CI does not need Python).
"""
from __future__ import annotations

import json
import sys

try:
    import numpy as np
    from numpy.linalg import norm
except ImportError:
    print("numpy required", file=sys.stderr)
    sys.exit(2)


def similarity_matrix(xq: np.ndarray, index: np.ndarray) -> np.ndarray:
    index_norm = norm(index, axis=1)
    xq_norm = norm(xq.T)
    return np.dot(index, xq.T) / (index_norm * xq_norm)


def main() -> None:
    index = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    xq = np.array([1.0, 0.0], dtype=np.float32)
    sim = similarity_matrix(xq, index)
    print(json.dumps({"sim": [float(x) for x in sim]}, indent=2))


if __name__ == "__main__":
    main()
