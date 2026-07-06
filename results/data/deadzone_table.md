## Matched pairs (ΔF reportable)

| pair | tier | V | corpus | bnd | F^BPE | F^UL | ΔF | flags |
|---|---|--:|---|---|--:|--:|--:|---|
| coconut__v1024_mb | headline | 1024 | coconut | mb | 0.5757 | 0.1678 | +0.4080 | BOTH-UNSAFE |
| coconut__v1024_nmb | headline | 1024 | coconut | nmb | 0.5665 | 0.1516 | +0.4149 | BOTH-UNSAFE |
| coconut__v256_mb | headline | 256 | coconut | mb | 0.9794 | 1.0000 | -0.0206 | ok |
| coconut__v256_nmb | headline | 256 | coconut | nmb | 1.0000 | 0.8969 | +0.1031 | one-unsafe |
| coconut__v512_mb | headline | 512 | coconut | mb | 0.9688 | 0.3654 | +0.6034 | one-unsafe |
| coconut__v512_nmb | headline | 512 | coconut | nmb | 0.9943 | 0.2720 | +0.7224 | one-unsafe |
| pubchem__v1024_mb | headline | 1024 | pubchem | mb | 0.9908 | 0.9792 | +0.0116 | ok |
| pubchem__v1024_mb__size_matched_700k | extras_size_matched | 1024 | pubchem | mb | 0.8994 | 0.2624 | +0.6370 | BOTH-UNSAFE |
| pubchem__v1024_nmb | headline | 1024 | pubchem | nmb | 1.0000 | 0.9572 | +0.0428 | ok |
| pubchem__v1024_nmb__size_matched_700k | extras_size_matched | 1024 | pubchem | nmb | 0.8185 | 0.1850 | +0.6335 | BOTH-UNSAFE |
| pubchem__v2048_mb | sensitivity | 2048 | pubchem | mb | 0.9799 | 0.6411 | +0.3388 | one-unsafe |
| pubchem__v2048_nmb | sensitivity | 2048 | pubchem | nmb | 0.9968 | 0.5675 | +0.4293 | one-unsafe |
| pubchem__v256_mb | headline | 256 | pubchem | mb | 0.9794 | 1.0000 | -0.0206 | ok |
| pubchem__v256_nmb | headline | 256 | pubchem | nmb | 1.0000 | 1.0000 | +0.0000 | ok |
| pubchem__v512_mb | headline | 512 | pubchem | mb | 0.9943 | 1.0000 | -0.0057 | ok |
| pubchem__v512_nmb | headline | 512 | pubchem | nmb | 1.0000 | 0.9972 | +0.0028 | ok |
| pubchem__v512_nmb__size_15m | extras_size_sweep | 512 | pubchem | nmb | 1.0000 | 0.9943 | +0.0057 | ok |
| pubchem__v512_nmb__size_5m | extras_size_sweep | 512 | pubchem | nmb | 1.0000 | 0.9207 | +0.0793 | one-unsafe |
| pubchem__v512_nmb__subsample_r1 | extras_subsample_redraw | 512 | pubchem | nmb | 1.0000 | 0.9207 | +0.0793 | one-unsafe |
| pubchem__v512_nmb__subsample_r2 | extras_subsample_redraw | 512 | pubchem | nmb | 1.0000 | 0.9207 | +0.0793 | one-unsafe |
| pubchem__v512_nmb__subsample_r3 | extras_subsample_redraw | 512 | pubchem | nmb | 1.0000 | 0.9122 | +0.0878 | one-unsafe |
| real_space__v1024_mb | anchor | 1024 | real_space | mb | 0.9931 | 0.7133 | +0.2798 | one-unsafe |
| real_space__v1024_nmb | anchor | 1024 | real_space | nmb | 1.0000 | 0.7041 | +0.2959 | one-unsafe |
| zinc22__v1024_mb | headline | 1024 | zinc22 | mb | 0.9838 | 0.5854 | +0.3984 | one-unsafe |
| zinc22__v1024_mb__size_matched_700k | extras_size_matched | 1024 | zinc22 | mb | 0.5376 | 0.1545 | +0.3831 | BOTH-UNSAFE |
| zinc22__v1024_nmb | headline | 1024 | zinc22 | nmb | 0.9988 | 0.5677 | +0.4311 | one-unsafe |
| zinc22__v1024_nmb__size_matched_700k | extras_size_matched | 1024 | zinc22 | nmb | 0.5364 | 0.1223 | +0.4141 | BOTH-UNSAFE |
| zinc22__v256_mb | headline | 256 | zinc22 | mb | 0.9588 | 1.0000 | -0.0412 | ok |
| zinc22__v256_nmb | headline | 256 | zinc22 | nmb | 1.0000 | 1.0000 | +0.0000 | ok |
| zinc22__v512_mb | headline | 512 | zinc22 | mb | 0.9773 | 0.5921 | +0.3853 | one-unsafe |
| zinc22__v512_nmb | headline | 512 | zinc22 | nmb | 1.0000 | 0.5677 | +0.4323 | one-unsafe |
| zinc22__v512_nmb__subsample_r1 | extras_subsample_redraw | 512 | zinc22 | nmb | 0.9972 | 0.5052 | +0.4919 | one-unsafe |
| zinc22__v512_nmb__subsample_r2 | extras_subsample_redraw | 512 | zinc22 | nmb | 0.9943 | 0.5000 | +0.4943 | one-unsafe |
| zinc22__v512_nmb__subsample_r3 | extras_subsample_redraw | 512 | zinc22 | nmb | 0.9943 | 0.4966 | +0.4978 | one-unsafe |

## Single-arm coordinates (ΔF undefined)

| pair | tier | V | corpus | bnd | arm | F | reason |
|---|---|--:|---|---|---|--:|---|
| pubchem__v1024_mb__seed_uncapped | extras_seed_cap | 1024 | pubchem | mb | unigram | 0.9792 | extras_single_arm_knob |
| pubchem__v256_mb__prune_shrink_0_9 | extras_prune_schedule | 256 | pubchem | mb | unigram | 1.0000 | extras_single_arm_knob |
| pubchem__v512_mb__prune_shrink_0_9 | extras_prune_schedule | 512 | pubchem | mb | unigram | 1.0000 | extras_single_arm_knob |
| real_space__v50000_nmb__merge_exhaustion | extras_merge_exhaustion | 50000 | real_space | nmb | bpe | 0.4671 | extras_single_arm_knob |
| zinc22__v2048_mb | conditional | 2048 | zinc22 | mb | bpe | 0.5246 | conditional_negative_branch |
| zinc22__v2048_nmb | conditional | 2048 | zinc22 | nmb | bpe | 0.5236 | conditional_negative_branch |
