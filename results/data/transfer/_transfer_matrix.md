# Transfer matrix (V=1024, nmb)

Off-diagonal = tokenizer trained on `train`, read on `eval`'s held-out split. Diagonal (native) reused from fertility. Penalty = off-domain fertility / native.


## bpe — fertility (penalty vs native)

train\eval | pubchem | zinc22 | coconut | real_space
---|---|---|---|---
pubchem | 36.12 (1.00x) | 32.31 (1.00x) | 54.93 (1.00x) | 32.75 (1.00x)
zinc22 | 36.18 (1.00x) | 32.16 (1.00x) | 55.21 (1.01x) | 32.72 (1.00x)
coconut | 36.01 (1.00x) | 32.35 (1.01x) | 54.88 (1.00x) | 32.82 (1.00x)
real_space | 36.18 (1.00x) | 32.28 (1.00x) | 55.28 (1.01x) | 32.71 (1.00x)

## unigram — fertility (penalty vs native)

train\eval | pubchem | zinc22 | coconut | real_space
---|---|---|---|---
pubchem | 50.95 (1.00x) | 44.41 (0.99x) | 75.49 (1.02x) | 45.75 (0.96x)
zinc22 | 51.02 (1.00x) | 44.97 (1.00x) | 76.25 (1.03x) | 46.95 (0.99x)
coconut | 47.78 (0.94x) | 43.17 (0.96x) | 74.17 (1.00x) | 44.69 (0.94x)
real_space | 53.54 (1.05x) | 45.52 (1.01x) | 79.75 (1.08x) | 47.42 (1.00x)
