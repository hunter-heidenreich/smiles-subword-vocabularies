## Matched pairs (Unigram entropy; BPE zero by construction)

| pair | tier | V | corpus | bnd | H/mol^UL | H/glyph^UL | BPE=0 |
|---|---|--:|---|---|--:|--:|:--:|
| coconut__v1024_mb | headline | 1024 | coconut | mb | 0.7179 | 0.0090 | ✓ |
| coconut__v1024_nmb | headline | 1024 | coconut | nmb | 0.7209 | 0.0090 | ✓ |
| coconut__v256_mb | headline | 256 | coconut | mb | 0.6841 | 0.0086 | ✓ |
| coconut__v256_nmb | headline | 256 | coconut | nmb | 0.6621 | 0.0083 | ✓ |
| coconut__v512_mb | headline | 512 | coconut | mb | 0.6950 | 0.0087 | ✓ |
| coconut__v512_nmb | headline | 512 | coconut | nmb | 0.6908 | 0.0087 | ✓ |
| pubchem__v1024_mb | headline | 1024 | pubchem | mb | 0.5271 | 0.0099 | ✓ |
| pubchem__v1024_mb__size_matched_700k | extras_size_matched | 1024 | pubchem | mb | 0.8372 | 0.0157 | ✓ |
| pubchem__v1024_nmb | headline | 1024 | pubchem | nmb | 0.4786 | 0.0090 | ✓ |
| pubchem__v1024_nmb__size_matched_700k | extras_size_matched | 1024 | pubchem | nmb | 0.8933 | 0.0167 | ✓ |
| pubchem__v2048_mb | sensitivity | 2048 | pubchem | mb | 0.5271 | 0.0099 | ✓ |
| pubchem__v2048_nmb | sensitivity | 2048 | pubchem | nmb | 0.4788 | 0.0090 | ✓ |
| pubchem__v256_mb | headline | 256 | pubchem | mb | 0.5081 | 0.0095 | ✓ |
| pubchem__v256_nmb | headline | 256 | pubchem | nmb | 0.4709 | 0.0088 | ✓ |
| pubchem__v512_mb | headline | 512 | pubchem | mb | 0.5254 | 0.0098 | ✓ |
| pubchem__v512_nmb | headline | 512 | pubchem | nmb | 0.4776 | 0.0089 | ✓ |
| pubchem__v512_nmb__size_15m | extras_size_sweep | 512 | pubchem | nmb | 0.4894 | 0.0092 | ✓ |
| pubchem__v512_nmb__size_5m | extras_size_sweep | 512 | pubchem | nmb | 0.5682 | 0.0106 | ✓ |
| pubchem__v512_nmb__subsample_r1 | extras_subsample_redraw | 512 | pubchem | nmb | 0.5668 | 0.0106 | ✓ |
| pubchem__v512_nmb__subsample_r2 | extras_subsample_redraw | 512 | pubchem | nmb | 0.5711 | 0.0107 | ✓ |
| pubchem__v512_nmb__subsample_r3 | extras_subsample_redraw | 512 | pubchem | nmb | 0.5625 | 0.0105 | ✓ |
| real_space__v1024_mb | anchor | 1024 | real_space | mb | 0.0304 | 0.0006 | ✓ |
| real_space__v1024_nmb | anchor | 1024 | real_space | nmb | 0.0314 | 0.0007 | ✓ |
| zinc22__v1024_mb | headline | 1024 | zinc22 | mb | 0.0311 | 0.0007 | ✓ |
| zinc22__v1024_mb__size_matched_700k | extras_size_matched | 1024 | zinc22 | mb | 1.4956 | 0.0328 | ✓ |
| zinc22__v1024_nmb | headline | 1024 | zinc22 | nmb | 0.0282 | 0.0006 | ✓ |
| zinc22__v1024_nmb__size_matched_700k | extras_size_matched | 1024 | zinc22 | nmb | 1.2331 | 0.0270 | ✓ |
| zinc22__v256_mb | headline | 256 | zinc22 | mb | 0.0260 | 0.0006 | ✓ |
| zinc22__v256_nmb | headline | 256 | zinc22 | nmb | 0.0239 | 0.0005 | ✓ |
| zinc22__v512_mb | headline | 512 | zinc22 | mb | 0.0309 | 0.0007 | ✓ |
| zinc22__v512_nmb | headline | 512 | zinc22 | nmb | 0.0282 | 0.0006 | ✓ |
| zinc22__v512_nmb__subsample_r1 | extras_subsample_redraw | 512 | zinc22 | nmb | 0.0308 | 0.0007 | ✓ |
| zinc22__v512_nmb__subsample_r2 | extras_subsample_redraw | 512 | zinc22 | nmb | 0.0316 | 0.0007 | ✓ |
| zinc22__v512_nmb__subsample_r3 | extras_subsample_redraw | 512 | zinc22 | nmb | 0.0314 | 0.0007 | ✓ |

## Single-arm coordinates

| pair | tier | V | corpus | bnd | arm | H/mol | H/glyph | reason |
|---|---|--:|---|---|---|--:|--:|---|
| pubchem__v1024_mb__seed_uncapped | extras_seed_cap | 1024 | pubchem | mb | unigram | 0.5271 | 0.0099 | extras_single_arm_knob |
| pubchem__v256_mb__prune_shrink_0_9 | extras_prune_schedule | 256 | pubchem | mb | unigram | 0.5074 | 0.0095 | extras_single_arm_knob |
| pubchem__v512_mb__prune_shrink_0_9 | extras_prune_schedule | 512 | pubchem | mb | unigram | 0.5252 | 0.0098 | extras_single_arm_knob |
| real_space__v50000_nmb__merge_exhaustion | extras_merge_exhaustion | 50000 | real_space | nmb | bpe | 0.0000 | 0.0000 | extras_single_arm_knob |
| zinc22__v2048_mb | conditional | 2048 | zinc22 | mb | bpe | 0.0000 | 0.0000 | conditional_negative_branch |
| zinc22__v2048_nmb | conditional | 2048 | zinc22 | nmb | bpe | 0.0000 | 0.0000 | conditional_negative_branch |
