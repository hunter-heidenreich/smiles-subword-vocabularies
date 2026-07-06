# Marginal cross-arm Jaccard per V-step

`fresh_arm = multi(arm, V_upper) \ multi(arm, V_lower)`; marginal J = Jaccard(fresh_bpe, fresh_unigram).

| corpus | bnd | step | fresh BPE | fresh UL | shared | marginal J |
|---|---|---|---|---|---|---|
| coconut | mb | 256->512 | 257 | 256 | 8 | 0.0158 |
| coconut | mb | 512->1024 | 512 | 416 | 12 | 0.0131 |
| coconut | nmb | 256->512 | 257 | 256 | 4 | 0.0079 |
| coconut | nmb | 512->1024 | 512 | 287 | 8 | 0.0101 |
| pubchem | mb | 256->512 | 256 | 256 | 8 | 0.0159 |
| pubchem | mb | 512->1024 | 512 | 512 | 4 | 0.0039 |
| pubchem | mb | 1024->2048 | 1024 | 1024 | 8 | 0.0039 |
| pubchem | nmb | 256->512 | 256 | 256 | 1 | 0.0020 |
| pubchem | nmb | 512->1024 | 512 | 512 | 0 | 0.0000 |
| pubchem | nmb | 1024->2048 | 1025 | 1024 | 6 | 0.0029 |
| zinc22 | mb | 256->512 | 257 | 256 | 2 | 0.0039 |
| zinc22 | mb | 512->1024 | 512 | 4 | 2 | 0.0039 |
| zinc22 | nmb | 256->512 | 257 | 213 | 3 | 0.0064 |
| zinc22 | nmb | 512->1024 | 512 | 0 | 0 | 0.0000 |
