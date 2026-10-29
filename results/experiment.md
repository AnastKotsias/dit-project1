# Image-search experiment

S=500, seed=1. Every vocabulary is trained on the same sampled training descriptors.

Validation uses D=10000 and 23 queries. For each approximate method, the configurations whose mAP@10 is within 0.01 of the method's best are treated as equally accurate, and the one with the fewest mean candidates (distance computations per query) is selected. The candidate count is used instead of the measured time so that the selection is reproducible.

Query times are single-threaded. Every configuration first runs the whole query set once untimed as a warm-up, then 5 timed passes over the query set; each query's time is the median of its 5 runs, and the reported time is the mean of these medians.

## Vocabulary comparison

Exact search on the validation set. The smallest K within 0.01 of the best mAP@10 is selected.

| K | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Selected |
|---:|---:|---:|---:|---:|:---:|
| 128 | 0.7217 | 0.7565 | 0.7264 | 0.7174 |  |
| 256 | 0.7304 | 0.7913 | 0.7551 | 1.2963 | yes |
| 512 | 0.7391 | 0.7826 | 0.7501 | 2.4483 |  |
| 1024 | 0.7391 | 0.7913 | 0.7436 | 4.8529 |  |

## K = 128

### Selected validation parameters

| Method | Parameters | Validation mAP@10 | Mean candidates |
|---|---|---:|---:|
| LSH | k=4, L=9, w=0.1 | 0.6784 | 2142.7 |
| HYPERCUBE | kproj=10, M=2000, w=0.05, probes=40 | 0.3774 | 1362.1 |
| IVFFLAT | kclusters=10, nprobe=2 | 0.7269 | 7531.0 |
| IVFPQ | kclusters=10, nprobe=2, M=16, nbits=8 | 0.6812 | 7531.0 |

### Validation results

Database: 10115 vectors (10000 MIRFlickr distractors).

#### EXACT (reference)

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| - | 0.7217 | 0.7565 | 0.7264 | 0.7174 | 10115.0 | 0/23 |  |

#### LSH

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| k=4, L=3, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0015 | 0.2 | 23/23 |  |
| k=4, L=5, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.3 | 23/23 |  |
| k=4, L=7, w=0.01 | 0.0087 | 0.0087 | 0.0087 | 0.0032 | 0.5 | 23/23 |  |
| k=4, L=9, w=0.01 | 0.0087 | 0.0087 | 0.0087 | 0.0041 | 0.5 | 23/23 |  |
| k=4, L=3, w=0.025 | 0.0522 | 0.0522 | 0.0522 | 0.0030 | 6.3 | 18/23 |  |
| k=4, L=5, w=0.025 | 0.0696 | 0.0696 | 0.0696 | 0.0043 | 11.9 | 9/23 |  |
| k=4, L=7, w=0.025 | 0.1304 | 0.1304 | 0.1304 | 0.0055 | 18.2 | 7/23 |  |
| k=4, L=9, w=0.025 | 0.1478 | 0.1478 | 0.1478 | 0.0066 | 21.1 | 6/23 |  |
| k=4, L=3, w=0.05 | 0.2348 | 0.2348 | 0.2290 | 0.0115 | 81.8 | 1/23 |  |
| k=4, L=5, w=0.05 | 0.2957 | 0.2957 | 0.2899 | 0.0269 | 153.7 | 0/23 |  |
| k=4, L=7, w=0.05 | 0.4174 | 0.4261 | 0.4130 | 0.0409 | 224.3 | 0/23 |  |
| k=4, L=9, w=0.05 | 0.4696 | 0.4783 | 0.4652 | 0.0500 | 263.3 | 0/23 |  |
| k=4, L=3, w=0.1 | 0.4000 | 0.4174 | 0.3983 | 0.1153 | 738.2 | 0/23 |  |
| k=4, L=5, w=0.1 | 0.6087 | 0.6261 | 0.6109 | 0.2105 | 1338.2 | 0/23 |  |
| k=4, L=7, w=0.1 | 0.6261 | 0.6609 | 0.6347 | 0.3105 | 1848.2 | 0/23 |  |
| k=4, L=9, w=0.1 | 0.6696 | 0.7043 | 0.6784 | 0.5620 | 2142.7 | 0/23 | yes |

#### HYPERCUBE

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| kproj=10, M=250, w=0.05, probes=5 | 0.1652 | 0.1652 | 0.1652 | 0.0300 | 186.2 | 0/23 |  |
| kproj=10, M=250, w=0.05, probes=10 | 0.1913 | 0.1913 | 0.1913 | 0.0369 | 219.5 | 0/23 |  |
| kproj=10, M=250, w=0.05, probes=20 | 0.2087 | 0.2087 | 0.2087 | 0.0432 | 247.7 | 0/23 |  |
| kproj=10, M=250, w=0.05, probes=40 | 0.2087 | 0.2087 | 0.2087 | 0.0431 | 250.0 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=5 | 0.1913 | 0.1913 | 0.1913 | 0.0389 | 247.7 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=10 | 0.2348 | 0.2348 | 0.2304 | 0.0585 | 353.3 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=20 | 0.2609 | 0.2696 | 0.2585 | 0.0744 | 446.2 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=40 | 0.2783 | 0.2870 | 0.2758 | 0.0843 | 492.2 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=5 | 0.1913 | 0.1913 | 0.1913 | 0.0387 | 250.4 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=10 | 0.2522 | 0.2609 | 0.2486 | 0.0736 | 441.2 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=20 | 0.2957 | 0.3043 | 0.2916 | 0.1139 | 741.0 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=40 | 0.3217 | 0.3304 | 0.3177 | 0.1437 | 890.8 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=5 | 0.1913 | 0.1913 | 0.1913 | 0.0390 | 250.4 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=10 | 0.2522 | 0.2609 | 0.2486 | 0.0731 | 441.2 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=20 | 0.2957 | 0.3130 | 0.2986 | 0.1304 | 861.8 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=40 | 0.3739 | 0.3913 | 0.3774 | 0.2281 | 1362.1 | 0/23 | yes |

#### IVFFLAT

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| kclusters=5, nprobe=1 | 0.5652 | 0.5826 | 0.5536 | 0.5609 | 5277.5 | 0/23 |  |
| kclusters=5, nprobe=2 | 0.7043 | 0.7478 | 0.7139 | 0.8973 | 8124.7 | 0/23 |  |
| kclusters=5, nprobe=3 | 0.7217 | 0.7565 | 0.7264 | 1.1465 | 9641.0 | 0/23 |  |
| kclusters=5, nprobe=5 | 0.7217 | 0.7565 | 0.7264 | 1.1486 | 10115.0 | 0/23 |  |
| kclusters=6, nprobe=1 | 0.5826 | 0.6000 | 0.5819 | 0.6002 | 5804.4 | 0/23 |  |
| kclusters=6, nprobe=2 | 0.7043 | 0.7304 | 0.7085 | 0.9059 | 8325.9 | 0/23 |  |
| kclusters=6, nprobe=3 | 0.7217 | 0.7565 | 0.7264 | 1.0717 | 9812.2 | 0/23 |  |
| kclusters=6, nprobe=5 | 0.7217 | 0.7565 | 0.7264 | 1.1051 | 10082.0 | 0/23 |  |
| kclusters=8, nprobe=1 | 0.6174 | 0.6435 | 0.6188 | 0.7013 | 5319.1 | 0/23 |  |
| kclusters=8, nprobe=2 | 0.7043 | 0.7391 | 0.7124 | 0.9111 | 7920.9 | 0/23 |  |
| kclusters=8, nprobe=3 | 0.7217 | 0.7565 | 0.7264 | 1.0752 | 9360.9 | 0/23 |  |
| kclusters=8, nprobe=5 | 0.7217 | 0.7565 | 0.7264 | 1.1668 | 9926.0 | 0/23 |  |
| kclusters=10, nprobe=1 | 0.6261 | 0.6522 | 0.6248 | 0.6154 | 5137.5 | 0/23 |  |
| kclusters=10, nprobe=2 | 0.7217 | 0.7565 | 0.7269 | 0.8876 | 7531.0 | 0/23 | yes |
| kclusters=10, nprobe=3 | 0.7217 | 0.7565 | 0.7264 | 1.0817 | 9170.9 | 0/23 |  |
| kclusters=10, nprobe=5 | 0.7217 | 0.7565 | 0.7264 | 1.1766 | 9740.0 | 0/23 |  |

#### IVFPQ

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| kclusters=5, nprobe=1, M=16, nbits=8 | 0.5130 | 0.5304 | 0.5154 | 0.0969 | 5277.5 | 0/23 |  |
| kclusters=5, nprobe=2, M=16, nbits=8 | 0.6435 | 0.6783 | 0.6509 | 0.1488 | 8124.7 | 0/23 |  |
| kclusters=5, nprobe=3, M=16, nbits=8 | 0.6435 | 0.6870 | 0.6538 | 0.1871 | 9641.0 | 0/23 |  |
| kclusters=5, nprobe=5, M=16, nbits=8 | 0.6435 | 0.6870 | 0.6538 | 0.2328 | 10115.0 | 0/23 |  |
| kclusters=6, nprobe=1, M=16, nbits=8 | 0.5130 | 0.5652 | 0.5380 | 0.0918 | 5804.4 | 0/23 |  |
| kclusters=6, nprobe=2, M=16, nbits=8 | 0.6087 | 0.6783 | 0.6363 | 0.1400 | 8325.9 | 0/23 |  |
| kclusters=6, nprobe=3, M=16, nbits=8 | 0.6261 | 0.7043 | 0.6511 | 0.1761 | 9812.2 | 0/23 |  |
| kclusters=6, nprobe=5, M=16, nbits=8 | 0.6261 | 0.7043 | 0.6511 | 0.2144 | 10082.0 | 0/23 |  |
| kclusters=8, nprobe=1, M=16, nbits=8 | 0.6000 | 0.6174 | 0.5940 | 0.0899 | 5319.1 | 0/23 |  |
| kclusters=8, nprobe=2, M=16, nbits=8 | 0.6783 | 0.7043 | 0.6706 | 0.1400 | 7920.9 | 0/23 |  |
| kclusters=8, nprobe=3, M=16, nbits=8 | 0.6783 | 0.7217 | 0.6797 | 0.1759 | 9360.9 | 0/23 |  |
| kclusters=8, nprobe=5, M=16, nbits=8 | 0.6783 | 0.7217 | 0.6797 | 0.2196 | 9926.0 | 0/23 |  |
| kclusters=10, nprobe=1, M=16, nbits=8 | 0.5913 | 0.6261 | 0.5960 | 0.0931 | 5137.5 | 0/23 |  |
| kclusters=10, nprobe=2, M=16, nbits=8 | 0.6783 | 0.7130 | 0.6812 | 0.1402 | 7531.0 | 0/23 | yes |
| kclusters=10, nprobe=3, M=16, nbits=8 | 0.6783 | 0.7130 | 0.6812 | 0.1788 | 9170.9 | 0/23 |  |
| kclusters=10, nprobe=5, M=16, nbits=8 | 0.6783 | 0.7130 | 0.6812 | 0.2240 | 9740.0 | 0/23 |  |

### Test results

#### D = 1000

Usable MIRFlickr distractors: 1000 (0 skipped without SIFT descriptors), database size: 1115

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7217 | 0.8000 | 0.7346 | 0.0386 | 1115.0 | 0/23 |
| Illumination | 0.7273 | 0.7636 | 0.7273 | 0.0386 | 1115.0 | 0/11 |
| Viewpoint | 0.7167 | 0.8333 | 0.7414 | 0.0387 | 1115.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6087 | 0.6609 | 0.6163 | 0.0212 | 210.9 | 1/23 |
| Illumination | 0.6000 | 0.6545 | 0.6000 | 0.0188 | 179.6 | 0/11 |
| Viewpoint | 0.6167 | 0.6667 | 0.6312 | 0.0235 | 239.6 | 1/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.4522 | 0.4522 | 0.4393 | 0.0113 | 143.9 | 0/23 |
| Illumination | 0.4000 | 0.4000 | 0.3955 | 0.0110 | 132.8 | 0/11 |
| Viewpoint | 0.5000 | 0.5000 | 0.4794 | 0.0116 | 154.1 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7217 | 0.8000 | 0.7346 | 0.0391 | 997.1 | 0/23 |
| Illumination | 0.7273 | 0.7636 | 0.7273 | 0.0407 | 1041.0 | 0/11 |
| Viewpoint | 0.7167 | 0.8333 | 0.7414 | 0.0376 | 956.8 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7130 | 0.7826 | 0.7249 | 0.0530 | 997.1 | 0/23 |
| Illumination | 0.7091 | 0.7455 | 0.7248 | 0.0537 | 1041.0 | 0/11 |
| Viewpoint | 0.7167 | 0.8167 | 0.7250 | 0.0524 | 956.8 | 0/12 |

#### D = 5000

Usable MIRFlickr distractors: 5000 (6 skipped without SIFT descriptors), database size: 5115

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6957 | 0.7478 | 0.7042 | 0.2985 | 5115.0 | 0/23 |
| Illumination | 0.7273 | 0.7455 | 0.7186 | 0.3017 | 5115.0 | 0/11 |
| Viewpoint | 0.6667 | 0.7500 | 0.6911 | 0.2955 | 5115.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.5826 | 0.6087 | 0.5878 | 0.1135 | 954.9 | 0/23 |
| Illumination | 0.5636 | 0.5818 | 0.5636 | 0.1035 | 837.9 | 0/11 |
| Viewpoint | 0.6000 | 0.6333 | 0.6100 | 0.1225 | 1062.1 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.4261 | 0.4522 | 0.4267 | 0.0638 | 651.7 | 0/23 |
| Illumination | 0.3818 | 0.4000 | 0.3909 | 0.0652 | 592.5 | 0/11 |
| Viewpoint | 0.4667 | 0.5000 | 0.4594 | 0.0626 | 706.0 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6783 | 0.7304 | 0.6868 | 0.3154 | 3515.0 | 0/23 |
| Illumination | 0.6909 | 0.7091 | 0.6822 | 0.3161 | 3455.5 | 0/11 |
| Viewpoint | 0.6667 | 0.7500 | 0.6911 | 0.3149 | 3569.5 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6522 | 0.7043 | 0.6559 | 0.0896 | 3515.0 | 0/23 |
| Illumination | 0.6727 | 0.6909 | 0.6545 | 0.0888 | 3455.5 | 0/11 |
| Viewpoint | 0.6333 | 0.7167 | 0.6572 | 0.0903 | 3569.5 | 0/12 |

#### D = 10000

Usable MIRFlickr distractors: 10000 (7 skipped without SIFT descriptors), database size: 10115

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6783 | 0.7130 | 0.6884 | 0.7950 | 10115.0 | 0/23 |
| Illumination | 0.7091 | 0.7273 | 0.7030 | 0.8228 | 10115.0 | 0/11 |
| Viewpoint | 0.6500 | 0.7000 | 0.6750 | 0.7694 | 10115.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.5739 | 0.6000 | 0.5826 | 0.3125 | 1833.4 | 0/23 |
| Illumination | 0.5636 | 0.5636 | 0.5545 | 0.2711 | 1618.6 | 0/11 |
| Viewpoint | 0.5833 | 0.6333 | 0.6083 | 0.3505 | 2030.3 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.4261 | 0.4435 | 0.4237 | 0.2028 | 1238.0 | 0/23 |
| Illumination | 0.3818 | 0.4000 | 0.3896 | 0.1927 | 1119.9 | 0/11 |
| Viewpoint | 0.4667 | 0.4833 | 0.4549 | 0.2119 | 1346.3 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6783 | 0.7217 | 0.6894 | 0.8817 | 7738.2 | 0/23 |
| Illumination | 0.7091 | 0.7273 | 0.7030 | 0.9285 | 8024.6 | 0/11 |
| Viewpoint | 0.6500 | 0.7167 | 0.6769 | 0.8387 | 7475.6 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6348 | 0.6870 | 0.6613 | 0.1464 | 7738.2 | 0/23 |
| Illumination | 0.6364 | 0.6909 | 0.6674 | 0.1518 | 8024.6 | 0/11 |
| Viewpoint | 0.6333 | 0.6833 | 0.6557 | 0.1414 | 7475.6 | 0/12 |

#### D = 25000

Usable MIRFlickr distractors: 24985 (15 skipped without SIFT descriptors), database size: 25100

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6522 | 0.7043 | 0.6713 | 1.6888 | 25100.0 | 0/23 |
| Illumination | 0.6545 | 0.7273 | 0.6764 | 1.7070 | 25100.0 | 0/11 |
| Viewpoint | 0.6500 | 0.6833 | 0.6667 | 1.6722 | 25100.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.5652 | 0.5913 | 0.5757 | 0.8954 | 4590.2 | 0/23 |
| Illumination | 0.5455 | 0.5636 | 0.5491 | 0.7992 | 4052.4 | 0/11 |
| Viewpoint | 0.5833 | 0.6167 | 0.6000 | 0.9836 | 5083.2 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.3130 | 0.3391 | 0.3214 | 0.4136 | 1834.3 | 0/23 |
| Illumination | 0.3273 | 0.3455 | 0.3333 | 0.4137 | 1833.5 | 0/11 |
| Viewpoint | 0.3000 | 0.3333 | 0.3106 | 0.4136 | 1835.1 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6435 | 0.6957 | 0.6626 | 2.4657 | 18052.7 | 0/23 |
| Illumination | 0.6545 | 0.7273 | 0.6764 | 2.5355 | 18515.3 | 0/11 |
| Viewpoint | 0.6333 | 0.6667 | 0.6500 | 2.4017 | 17628.7 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.5826 | 0.6261 | 0.5941 | 0.2735 | 18052.7 | 0/23 |
| Illumination | 0.5818 | 0.6545 | 0.6072 | 0.2812 | 18515.3 | 0/11 |
| Viewpoint | 0.5833 | 0.6000 | 0.5821 | 0.2664 | 17628.7 | 0/12 |

## K = 256

### Selected validation parameters

| Method | Parameters | Validation mAP@10 | Mean candidates |
|---|---|---:|---:|
| LSH | k=4, L=7, w=0.1 | 0.7148 | 2044.5 |
| HYPERCUBE | kproj=10, M=2000, w=0.05, probes=40 | 0.3580 | 1063.2 |
| IVFFLAT | kclusters=6, nprobe=2 | 0.7522 | 8425.0 |
| IVFPQ | kclusters=6, nprobe=2, M=16, nbits=8 | 0.6761 | 8425.0 |

### Validation results

Database: 10115 vectors (10000 MIRFlickr distractors).

#### EXACT (reference)

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| - | 0.7304 | 0.7913 | 0.7551 | 1.2963 | 10115.0 | 0/23 |  |

#### LSH

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| k=4, L=3, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0027 | 0.2 | 23/23 |  |
| k=4, L=5, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0042 | 0.2 | 23/23 |  |
| k=4, L=7, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0058 | 0.4 | 23/23 |  |
| k=4, L=9, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0075 | 0.5 | 23/23 |  |
| k=4, L=3, w=0.025 | 0.0261 | 0.0261 | 0.0261 | 0.0045 | 9.4 | 13/23 |  |
| k=4, L=5, w=0.025 | 0.0522 | 0.0522 | 0.0522 | 0.0062 | 14.1 | 9/23 |  |
| k=4, L=7, w=0.025 | 0.0783 | 0.0783 | 0.0783 | 0.0087 | 20.5 | 6/23 |  |
| k=4, L=9, w=0.025 | 0.0957 | 0.0957 | 0.0957 | 0.0115 | 27.3 | 5/23 |  |
| k=4, L=3, w=0.05 | 0.2696 | 0.2696 | 0.2674 | 0.0400 | 135.0 | 2/23 |  |
| k=4, L=5, w=0.05 | 0.3652 | 0.3652 | 0.3630 | 0.0593 | 195.6 | 2/23 |  |
| k=4, L=7, w=0.05 | 0.4261 | 0.4261 | 0.4226 | 0.0849 | 275.5 | 1/23 |  |
| k=4, L=9, w=0.05 | 0.4957 | 0.5043 | 0.5000 | 0.1103 | 374.2 | 1/23 |  |
| k=4, L=3, w=0.1 | 0.5565 | 0.5739 | 0.5604 | 0.3771 | 1043.4 | 0/23 |  |
| k=4, L=5, w=0.1 | 0.6609 | 0.6783 | 0.6546 | 0.4181 | 1504.4 | 0/23 |  |
| k=4, L=7, w=0.1 | 0.7043 | 0.7391 | 0.7148 | 0.5601 | 2044.5 | 0/23 | yes |
| k=4, L=9, w=0.1 | 0.7130 | 0.7565 | 0.7224 | 0.7151 | 2634.4 | 0/23 |  |

#### HYPERCUBE

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| kproj=10, M=250, w=0.05, probes=5 | 0.1478 | 0.1478 | 0.1435 | 0.0415 | 154.5 | 0/23 |  |
| kproj=10, M=250, w=0.05, probes=10 | 0.1739 | 0.1739 | 0.1696 | 0.0565 | 199.6 | 0/23 |  |
| kproj=10, M=250, w=0.05, probes=20 | 0.1739 | 0.1739 | 0.1696 | 0.0701 | 236.0 | 0/23 |  |
| kproj=10, M=250, w=0.05, probes=40 | 0.1739 | 0.1739 | 0.1696 | 0.1329 | 247.6 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=5 | 0.1565 | 0.1565 | 0.1522 | 0.0476 | 171.9 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=10 | 0.2261 | 0.2261 | 0.2188 | 0.0854 | 309.2 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=20 | 0.2783 | 0.2783 | 0.2722 | 0.1158 | 407.7 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=40 | 0.2783 | 0.2783 | 0.2722 | 0.1376 | 470.7 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=5 | 0.1565 | 0.1565 | 0.1522 | 0.0451 | 171.9 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=10 | 0.2348 | 0.2348 | 0.2275 | 0.0940 | 345.3 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=20 | 0.2957 | 0.3043 | 0.2935 | 0.1795 | 615.5 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=40 | 0.3478 | 0.3565 | 0.3452 | 0.2393 | 791.6 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=5 | 0.1565 | 0.1565 | 0.1522 | 0.0452 | 171.9 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=10 | 0.2348 | 0.2348 | 0.2275 | 0.0935 | 345.3 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=20 | 0.3130 | 0.3217 | 0.3045 | 0.1922 | 661.7 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=40 | 0.3652 | 0.3826 | 0.3580 | 0.3319 | 1063.2 | 0/23 | yes |

#### IVFFLAT

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| kclusters=5, nprobe=1 | 0.6000 | 0.6435 | 0.6159 | 1.0359 | 6655.1 | 0/23 |  |
| kclusters=5, nprobe=2 | 0.7304 | 0.7913 | 0.7551 | 1.6318 | 9889.3 | 0/23 |  |
| kclusters=5, nprobe=3 | 0.7304 | 0.7913 | 0.7551 | 1.7087 | 9989.7 | 0/23 |  |
| kclusters=5, nprobe=5 | 0.7304 | 0.7913 | 0.7551 | 1.7221 | 10115.0 | 0/23 |  |
| kclusters=6, nprobe=1 | 0.6522 | 0.6870 | 0.6652 | 1.0340 | 6272.3 | 0/23 |  |
| kclusters=6, nprobe=2 | 0.7304 | 0.7826 | 0.7522 | 1.4934 | 8425.0 | 0/23 | yes |
| kclusters=6, nprobe=3 | 0.7304 | 0.7913 | 0.7551 | 1.7966 | 9886.0 | 0/23 |  |
| kclusters=6, nprobe=5 | 0.7304 | 0.7913 | 0.7551 | 1.8826 | 10086.0 | 0/23 |  |
| kclusters=8, nprobe=1 | 0.7043 | 0.7652 | 0.7279 | 1.1561 | 7694.6 | 0/23 |  |
| kclusters=8, nprobe=2 | 0.7304 | 0.7913 | 0.7551 | 1.4397 | 8975.8 | 0/23 |  |
| kclusters=8, nprobe=3 | 0.7304 | 0.7913 | 0.7551 | 1.6746 | 9850.0 | 0/23 |  |
| kclusters=8, nprobe=5 | 0.7304 | 0.7913 | 0.7551 | 1.6688 | 9958.1 | 0/23 |  |
| kclusters=10, nprobe=1 | 0.7043 | 0.7565 | 0.7253 | 1.2129 | 7891.8 | 0/23 |  |
| kclusters=10, nprobe=2 | 0.7304 | 0.7913 | 0.7551 | 1.4732 | 9253.2 | 0/23 |  |
| kclusters=10, nprobe=3 | 0.7304 | 0.7913 | 0.7551 | 1.5120 | 9686.3 | 0/23 |  |
| kclusters=10, nprobe=5 | 0.7304 | 0.7913 | 0.7551 | 1.5736 | 9766.0 | 0/23 |  |

#### IVFPQ

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| kclusters=5, nprobe=1, M=16, nbits=8 | 0.5391 | 0.5913 | 0.5487 | 0.1115 | 6655.1 | 0/23 |  |
| kclusters=5, nprobe=2, M=16, nbits=8 | 0.6348 | 0.7043 | 0.6483 | 0.1719 | 9889.3 | 0/23 |  |
| kclusters=5, nprobe=3, M=16, nbits=8 | 0.6348 | 0.7043 | 0.6483 | 0.1981 | 9989.7 | 0/23 |  |
| kclusters=5, nprobe=5, M=16, nbits=8 | 0.6348 | 0.7043 | 0.6483 | 0.2490 | 10115.0 | 0/23 |  |
| kclusters=6, nprobe=1, M=16, nbits=8 | 0.6087 | 0.6348 | 0.6104 | 0.1041 | 6272.3 | 0/23 |  |
| kclusters=6, nprobe=2, M=16, nbits=8 | 0.6696 | 0.7130 | 0.6761 | 0.1533 | 8425.0 | 0/23 | yes |
| kclusters=6, nprobe=3, M=16, nbits=8 | 0.6696 | 0.7130 | 0.6761 | 0.1961 | 9886.0 | 0/23 |  |
| kclusters=6, nprobe=5, M=16, nbits=8 | 0.6696 | 0.7130 | 0.6761 | 0.2482 | 10086.0 | 0/23 |  |
| kclusters=8, nprobe=1, M=16, nbits=8 | 0.6348 | 0.6783 | 0.6546 | 0.1227 | 7694.6 | 0/23 |  |
| kclusters=8, nprobe=2, M=16, nbits=8 | 0.6522 | 0.6957 | 0.6676 | 0.1589 | 8975.8 | 0/23 |  |
| kclusters=8, nprobe=3, M=16, nbits=8 | 0.6522 | 0.6957 | 0.6676 | 0.1971 | 9850.0 | 0/23 |  |
| kclusters=8, nprobe=5, M=16, nbits=8 | 0.6522 | 0.6957 | 0.6676 | 0.2569 | 9958.1 | 0/23 |  |
| kclusters=10, nprobe=1, M=16, nbits=8 | 0.6435 | 0.6957 | 0.6582 | 0.1253 | 7891.8 | 0/23 |  |
| kclusters=10, nprobe=2, M=16, nbits=8 | 0.6783 | 0.7304 | 0.6837 | 0.1622 | 9253.2 | 0/23 |  |
| kclusters=10, nprobe=3, M=16, nbits=8 | 0.6783 | 0.7304 | 0.6837 | 0.1931 | 9686.3 | 0/23 |  |
| kclusters=10, nprobe=5, M=16, nbits=8 | 0.6783 | 0.7304 | 0.6837 | 0.2450 | 9766.0 | 0/23 |  |

### Test results

#### D = 1000

Usable MIRFlickr distractors: 1000 (0 skipped without SIFT descriptors), database size: 1115

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7391 | 0.8087 | 0.7551 | 0.0971 | 1115.0 | 0/23 |
| Illumination | 0.7455 | 0.7455 | 0.7345 | 0.1133 | 1115.0 | 0/11 |
| Viewpoint | 0.7333 | 0.8667 | 0.7739 | 0.0822 | 1115.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6000 | 0.6609 | 0.6278 | 0.0303 | 237.3 | 0/23 |
| Illumination | 0.5818 | 0.6364 | 0.6059 | 0.0292 | 229.7 | 0/11 |
| Viewpoint | 0.6167 | 0.6833 | 0.6479 | 0.0313 | 244.2 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.3739 | 0.4000 | 0.3693 | 0.0153 | 133.0 | 0/23 |
| Illumination | 0.3636 | 0.4000 | 0.3646 | 0.0157 | 138.4 | 0/11 |
| Viewpoint | 0.3833 | 0.4000 | 0.3736 | 0.0149 | 128.0 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7391 | 0.8087 | 0.7551 | 0.0642 | 1059.7 | 0/23 |
| Illumination | 0.7455 | 0.7455 | 0.7345 | 0.0643 | 1061.5 | 0/11 |
| Viewpoint | 0.7333 | 0.8667 | 0.7739 | 0.0641 | 1058.2 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7043 | 0.8000 | 0.7306 | 0.0651 | 1059.7 | 0/23 |
| Illumination | 0.7273 | 0.7636 | 0.7364 | 0.0651 | 1061.5 | 0/11 |
| Viewpoint | 0.6833 | 0.8333 | 0.7253 | 0.0651 | 1058.2 | 0/12 |

#### D = 5000

Usable MIRFlickr distractors: 5000 (6 skipped without SIFT descriptors), database size: 5115

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7130 | 0.7565 | 0.7171 | 0.6581 | 5115.0 | 0/23 |
| Illumination | 0.7091 | 0.7273 | 0.7115 | 0.6442 | 5115.0 | 0/11 |
| Viewpoint | 0.7167 | 0.7833 | 0.7222 | 0.6709 | 5115.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6000 | 0.6261 | 0.6116 | 0.2570 | 1075.4 | 0/23 |
| Illumination | 0.5818 | 0.5818 | 0.5818 | 0.2496 | 1049.1 | 0/11 |
| Viewpoint | 0.6167 | 0.6667 | 0.6388 | 0.2638 | 1099.5 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.3739 | 0.3913 | 0.3639 | 0.1413 | 585.4 | 0/23 |
| Illumination | 0.3636 | 0.3818 | 0.3543 | 0.1401 | 609.4 | 0/11 |
| Viewpoint | 0.3833 | 0.4000 | 0.3728 | 0.1424 | 563.4 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7130 | 0.7565 | 0.7171 | 0.8461 | 4795.3 | 0/23 |
| Illumination | 0.7091 | 0.7273 | 0.7115 | 0.7714 | 4794.1 | 0/11 |
| Viewpoint | 0.7167 | 0.7833 | 0.7222 | 0.9146 | 4796.3 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6696 | 0.7304 | 0.6812 | 0.1119 | 4795.3 | 0/23 |
| Illumination | 0.6909 | 0.7273 | 0.6921 | 0.1124 | 4794.1 | 0/11 |
| Viewpoint | 0.6500 | 0.7333 | 0.6711 | 0.1114 | 4796.3 | 0/12 |

#### D = 10000

Usable MIRFlickr distractors: 10000 (7 skipped without SIFT descriptors), database size: 10115

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7130 | 0.7478 | 0.7113 | 1.2656 | 10115.0 | 0/23 |
| Illumination | 0.7091 | 0.7091 | 0.7055 | 1.2677 | 10115.0 | 0/11 |
| Viewpoint | 0.7167 | 0.7833 | 0.7167 | 1.2636 | 10115.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6000 | 0.6261 | 0.6116 | 0.5718 | 2090.4 | 0/23 |
| Illumination | 0.5818 | 0.5818 | 0.5818 | 0.5618 | 2045.9 | 0/11 |
| Viewpoint | 0.6167 | 0.6667 | 0.6388 | 0.5810 | 2131.2 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.3652 | 0.3739 | 0.3593 | 0.3496 | 1136.0 | 0/23 |
| Illumination | 0.3455 | 0.3636 | 0.3477 | 0.3634 | 1189.3 | 0/11 |
| Viewpoint | 0.3833 | 0.3833 | 0.3700 | 0.3370 | 1087.2 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7130 | 0.7478 | 0.7113 | 1.4978 | 8578.2 | 0/23 |
| Illumination | 0.7091 | 0.7091 | 0.7055 | 1.5662 | 8941.6 | 0/11 |
| Viewpoint | 0.7167 | 0.7833 | 0.7167 | 1.4351 | 8245.1 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6261 | 0.7043 | 0.6562 | 0.1773 | 8578.2 | 0/23 |
| Illumination | 0.6545 | 0.6909 | 0.6737 | 0.1826 | 8941.6 | 0/11 |
| Viewpoint | 0.6000 | 0.7167 | 0.6401 | 0.1725 | 8245.1 | 0/12 |

#### D = 25000

Usable MIRFlickr distractors: 24985 (15 skipped without SIFT descriptors), database size: 25100

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6957 | 0.7478 | 0.7046 | 3.2699 | 25100.0 | 0/23 |
| Illumination | 0.7091 | 0.7091 | 0.7055 | 3.2768 | 25100.0 | 0/11 |
| Viewpoint | 0.6833 | 0.7833 | 0.7038 | 3.2636 | 25100.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6000 | 0.6261 | 0.6116 | 1.4591 | 5283.8 | 0/23 |
| Illumination | 0.5818 | 0.5818 | 0.5818 | 1.4317 | 5162.3 | 0/11 |
| Viewpoint | 0.6167 | 0.6667 | 0.6388 | 1.4842 | 5395.2 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.2870 | 0.2957 | 0.2812 | 0.7553 | 1738.2 | 0/23 |
| Illumination | 0.2727 | 0.2909 | 0.2753 | 0.7328 | 1786.3 | 0/11 |
| Viewpoint | 0.3000 | 0.3000 | 0.2867 | 0.7759 | 1694.1 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6957 | 0.7478 | 0.7046 | 4.2991 | 21732.5 | 0/23 |
| Illumination | 0.7091 | 0.7091 | 0.7055 | 4.2089 | 21497.1 | 0/11 |
| Viewpoint | 0.6833 | 0.7833 | 0.7038 | 4.3817 | 21948.2 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6087 | 0.6696 | 0.6124 | 0.3216 | 21732.5 | 0/23 |
| Illumination | 0.6000 | 0.6727 | 0.6067 | 0.3200 | 21497.1 | 0/11 |
| Viewpoint | 0.6167 | 0.6667 | 0.6176 | 0.3230 | 21948.2 | 0/12 |

## K = 512

### Selected validation parameters

| Method | Parameters | Validation mAP@10 | Mean candidates |
|---|---|---:|---:|
| LSH | k=4, L=9, w=0.1 | 0.6772 | 3880.0 |
| HYPERCUBE | kproj=10, M=2000, w=0.05, probes=40 | 0.4715 | 1681.2 |
| IVFFLAT | kclusters=10, nprobe=2 | 0.7529 | 9199.6 |
| IVFPQ | kclusters=6, nprobe=2, M=16, nbits=8 | 0.6456 | 9229.8 |

### Validation results

Database: 10115 vectors (10000 MIRFlickr distractors).

#### EXACT (reference)

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| - | 0.7391 | 0.7826 | 0.7501 | 2.4483 | 10115.0 | 0/23 |  |

#### LSH

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| k=4, L=3, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0051 | 0.5 | 23/23 |  |
| k=4, L=5, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0083 | 0.8 | 23/23 |  |
| k=4, L=7, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0114 | 0.9 | 23/23 |  |
| k=4, L=9, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0145 | 1.1 | 23/23 |  |
| k=4, L=3, w=0.025 | 0.0435 | 0.0435 | 0.0435 | 0.0091 | 19.5 | 7/23 |  |
| k=4, L=5, w=0.025 | 0.0957 | 0.0957 | 0.0957 | 0.0384 | 33.1 | 5/23 |  |
| k=4, L=7, w=0.025 | 0.1391 | 0.1391 | 0.1391 | 0.0331 | 41.6 | 2/23 |  |
| k=4, L=9, w=0.025 | 0.1478 | 0.1478 | 0.1435 | 0.0426 | 50.3 | 2/23 |  |
| k=4, L=3, w=0.05 | 0.2870 | 0.2957 | 0.2849 | 0.1101 | 226.9 | 1/23 |  |
| k=4, L=5, w=0.05 | 0.4000 | 0.4000 | 0.3949 | 0.1917 | 388.1 | 1/23 |  |
| k=4, L=7, w=0.05 | 0.4696 | 0.4696 | 0.4632 | 0.2406 | 489.8 | 0/23 |  |
| k=4, L=9, w=0.05 | 0.5130 | 0.5130 | 0.5067 | 0.3071 | 593.3 | 0/23 |  |
| k=4, L=3, w=0.1 | 0.5304 | 0.5478 | 0.5284 | 0.7695 | 1830.3 | 0/23 |  |
| k=4, L=5, w=0.1 | 0.6261 | 0.6522 | 0.6275 | 1.1470 | 2809.1 | 0/23 |  |
| k=4, L=7, w=0.1 | 0.6609 | 0.6870 | 0.6603 | 1.3604 | 3397.0 | 0/23 |  |
| k=4, L=9, w=0.1 | 0.6696 | 0.7217 | 0.6772 | 1.5457 | 3880.0 | 0/23 | yes |

#### HYPERCUBE

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| kproj=10, M=250, w=0.05, probes=5 | 0.1739 | 0.1826 | 0.1704 | 0.1049 | 208.5 | 0/23 |  |
| kproj=10, M=250, w=0.05, probes=10 | 0.1826 | 0.1913 | 0.1791 | 0.1216 | 239.0 | 0/23 |  |
| kproj=10, M=250, w=0.05, probes=20 | 0.1826 | 0.1913 | 0.1791 | 0.1277 | 247.6 | 0/23 |  |
| kproj=10, M=250, w=0.05, probes=40 | 0.1826 | 0.1913 | 0.1791 | 0.1348 | 250.0 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=5 | 0.1826 | 0.1826 | 0.1783 | 0.1496 | 307.3 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=10 | 0.2957 | 0.3043 | 0.2928 | 0.2187 | 429.6 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=20 | 0.2957 | 0.3043 | 0.2928 | 0.2387 | 479.2 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=40 | 0.2957 | 0.3043 | 0.2928 | 0.2525 | 500.0 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=5 | 0.1826 | 0.1826 | 0.1783 | 0.1731 | 350.6 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=10 | 0.3391 | 0.3565 | 0.3421 | 0.3499 | 655.7 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=20 | 0.3739 | 0.3826 | 0.3703 | 0.4402 | 868.7 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=40 | 0.3913 | 0.4000 | 0.3877 | 0.4995 | 971.9 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=5 | 0.1826 | 0.1826 | 0.1783 | 0.1727 | 350.6 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=10 | 0.3565 | 0.3739 | 0.3526 | 0.3797 | 746.6 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=20 | 0.4348 | 0.4522 | 0.4318 | 0.6812 | 1230.3 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=40 | 0.4783 | 0.4957 | 0.4715 | 0.8582 | 1681.2 | 0/23 | yes |

#### IVFFLAT

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| kclusters=5, nprobe=1 | 0.7043 | 0.7565 | 0.7181 | 2.3560 | 9037.5 | 0/23 |  |
| kclusters=5, nprobe=2 | 0.7304 | 0.7739 | 0.7414 | 2.4532 | 9268.2 | 0/23 |  |
| kclusters=5, nprobe=3 | 0.7391 | 0.7826 | 0.7501 | 2.6692 | 9859.2 | 0/23 |  |
| kclusters=5, nprobe=5 | 0.7391 | 0.7826 | 0.7501 | 2.8148 | 10115.0 | 0/23 |  |
| kclusters=6, nprobe=1 | 0.7217 | 0.7739 | 0.7355 | 2.3736 | 9058.5 | 0/23 |  |
| kclusters=6, nprobe=2 | 0.7391 | 0.7826 | 0.7501 | 2.4627 | 9229.8 | 0/23 |  |
| kclusters=6, nprobe=3 | 0.7391 | 0.7826 | 0.7501 | 2.7094 | 9838.2 | 0/23 |  |
| kclusters=6, nprobe=5 | 0.7391 | 0.7826 | 0.7501 | 2.7666 | 10040.4 | 0/23 |  |
| kclusters=8, nprobe=1 | 0.7130 | 0.7565 | 0.7225 | 2.1565 | 8049.8 | 0/23 |  |
| kclusters=8, nprobe=2 | 0.7391 | 0.7913 | 0.7529 | 2.5833 | 9336.2 | 0/23 |  |
| kclusters=8, nprobe=3 | 0.7391 | 0.7826 | 0.7501 | 2.7737 | 9862.2 | 0/23 |  |
| kclusters=8, nprobe=5 | 0.7391 | 0.7826 | 0.7501 | 2.8142 | 9978.7 | 0/23 |  |
| kclusters=10, nprobe=1 | 0.7130 | 0.7565 | 0.7225 | 2.1115 | 7943.7 | 0/23 |  |
| kclusters=10, nprobe=2 | 0.7391 | 0.7913 | 0.7529 | 2.5522 | 9199.6 | 0/23 | yes |
| kclusters=10, nprobe=3 | 0.7391 | 0.7826 | 0.7501 | 2.7846 | 9842.9 | 0/23 |  |
| kclusters=10, nprobe=5 | 0.7391 | 0.7826 | 0.7501 | 2.8574 | 9972.0 | 0/23 |  |

#### IVFPQ

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| kclusters=5, nprobe=1, M=16, nbits=8 | 0.6087 | 0.6522 | 0.6158 | 0.1558 | 9037.5 | 0/23 |  |
| kclusters=5, nprobe=2, M=16, nbits=8 | 0.6261 | 0.6696 | 0.6332 | 0.1945 | 9268.2 | 0/23 |  |
| kclusters=5, nprobe=3, M=16, nbits=8 | 0.6348 | 0.6783 | 0.6376 | 0.2409 | 9859.2 | 0/23 |  |
| kclusters=5, nprobe=5, M=16, nbits=8 | 0.6348 | 0.6783 | 0.6376 | 0.3252 | 10115.0 | 0/23 |  |
| kclusters=6, nprobe=1, M=16, nbits=8 | 0.6522 | 0.6783 | 0.6347 | 0.1563 | 9058.5 | 0/23 |  |
| kclusters=6, nprobe=2, M=16, nbits=8 | 0.6696 | 0.6957 | 0.6456 | 0.1933 | 9229.8 | 0/23 | yes |
| kclusters=6, nprobe=3, M=16, nbits=8 | 0.6696 | 0.6957 | 0.6456 | 0.2417 | 9838.2 | 0/23 |  |
| kclusters=6, nprobe=5, M=16, nbits=8 | 0.6696 | 0.6957 | 0.6456 | 0.3252 | 10040.4 | 0/23 |  |
| kclusters=8, nprobe=1, M=16, nbits=8 | 0.6174 | 0.6696 | 0.6169 | 0.1446 | 8049.8 | 0/23 |  |
| kclusters=8, nprobe=2, M=16, nbits=8 | 0.6435 | 0.7043 | 0.6423 | 0.1953 | 9336.2 | 0/23 |  |
| kclusters=8, nprobe=3, M=16, nbits=8 | 0.6435 | 0.7043 | 0.6423 | 0.3180 | 9862.2 | 0/23 |  |
| kclusters=8, nprobe=5, M=16, nbits=8 | 0.6435 | 0.7043 | 0.6423 | 0.3346 | 9978.7 | 0/23 |  |
| kclusters=10, nprobe=1, M=16, nbits=8 | 0.6087 | 0.6522 | 0.6066 | 0.1565 | 7943.7 | 0/23 |  |
| kclusters=10, nprobe=2, M=16, nbits=8 | 0.6348 | 0.6783 | 0.6239 | 0.2049 | 9199.6 | 0/23 |  |
| kclusters=10, nprobe=3, M=16, nbits=8 | 0.6348 | 0.6783 | 0.6239 | 0.2506 | 9842.9 | 0/23 |  |
| kclusters=10, nprobe=5, M=16, nbits=8 | 0.6348 | 0.6783 | 0.6239 | 0.3301 | 9972.0 | 0/23 |  |

### Test results

#### D = 1000

Usable MIRFlickr distractors: 1000 (0 skipped without SIFT descriptors), database size: 1115

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7130 | 0.7652 | 0.7161 | 0.2233 | 1115.0 | 0/23 |
| Illumination | 0.7091 | 0.7091 | 0.6848 | 0.2232 | 1115.0 | 0/11 |
| Viewpoint | 0.7167 | 0.8167 | 0.7448 | 0.2233 | 1115.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6522 | 0.6870 | 0.6569 | 0.1056 | 408.4 | 0/23 |
| Illumination | 0.6727 | 0.6727 | 0.6485 | 0.1022 | 384.0 | 0/11 |
| Viewpoint | 0.6333 | 0.7000 | 0.6646 | 0.1087 | 430.8 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.4435 | 0.4609 | 0.4328 | 0.0484 | 244.4 | 0/23 |
| Illumination | 0.4727 | 0.4727 | 0.4400 | 0.0469 | 232.8 | 0/11 |
| Viewpoint | 0.4167 | 0.4500 | 0.4262 | 0.0499 | 255.1 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7130 | 0.7652 | 0.7161 | 0.2241 | 1061.2 | 0/23 |
| Illumination | 0.7091 | 0.7091 | 0.6848 | 0.2211 | 1064.1 | 0/11 |
| Viewpoint | 0.7167 | 0.8167 | 0.7448 | 0.2268 | 1058.5 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6870 | 0.7652 | 0.6914 | 0.0968 | 1079.9 | 0/23 |
| Illumination | 0.7273 | 0.7455 | 0.7200 | 0.0969 | 1083.3 | 0/11 |
| Viewpoint | 0.6500 | 0.7833 | 0.6651 | 0.0966 | 1076.8 | 0/12 |

#### D = 5000

Usable MIRFlickr distractors: 5000 (6 skipped without SIFT descriptors), database size: 5115

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6957 | 0.7478 | 0.6975 | 1.2676 | 5115.0 | 0/23 |
| Illumination | 0.7091 | 0.7091 | 0.6818 | 1.2671 | 5115.0 | 0/11 |
| Viewpoint | 0.6833 | 0.7833 | 0.7118 | 1.2680 | 5115.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6522 | 0.6870 | 0.6554 | 0.7389 | 1838.0 | 0/23 |
| Illumination | 0.6727 | 0.6727 | 0.6455 | 0.6941 | 1711.2 | 0/11 |
| Viewpoint | 0.6333 | 0.7000 | 0.6646 | 0.7799 | 1954.2 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.4348 | 0.4522 | 0.4248 | 0.5065 | 1059.0 | 0/23 |
| Illumination | 0.4545 | 0.4545 | 0.4288 | 0.4813 | 1005.5 | 0/11 |
| Viewpoint | 0.4167 | 0.4500 | 0.4212 | 0.5296 | 1108.1 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6957 | 0.7478 | 0.6975 | 1.2922 | 4834.0 | 0/23 |
| Illumination | 0.7091 | 0.7091 | 0.6818 | 1.2937 | 4834.2 | 0/11 |
| Viewpoint | 0.6833 | 0.7833 | 0.7118 | 1.2908 | 4833.9 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6348 | 0.6870 | 0.6379 | 0.1595 | 5070.0 | 0/23 |
| Illumination | 0.6727 | 0.7091 | 0.6721 | 0.1595 | 5070.0 | 0/11 |
| Viewpoint | 0.6000 | 0.6667 | 0.6065 | 0.1595 | 5070.0 | 0/12 |

#### D = 10000

Usable MIRFlickr distractors: 10000 (7 skipped without SIFT descriptors), database size: 10115

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6957 | 0.7478 | 0.6958 | 2.5407 | 10115.0 | 0/23 |
| Illumination | 0.7091 | 0.7091 | 0.6818 | 2.5514 | 10115.0 | 0/11 |
| Viewpoint | 0.6833 | 0.7833 | 0.7087 | 2.5309 | 10115.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6522 | 0.6870 | 0.6554 | 1.4327 | 3578.6 | 0/23 |
| Illumination | 0.6727 | 0.6727 | 0.6455 | 1.3438 | 3322.2 | 0/11 |
| Viewpoint | 0.6333 | 0.7000 | 0.6646 | 1.5142 | 3813.7 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.3913 | 0.4087 | 0.3813 | 0.8855 | 1711.6 | 0/23 |
| Illumination | 0.4364 | 0.4364 | 0.4106 | 0.8483 | 1636.4 | 0/11 |
| Viewpoint | 0.3500 | 0.3833 | 0.3545 | 0.9196 | 1780.6 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6957 | 0.7478 | 0.6960 | 2.5707 | 9272.2 | 0/23 |
| Illumination | 0.7091 | 0.7091 | 0.6818 | 2.5879 | 9313.1 | 0/11 |
| Viewpoint | 0.6833 | 0.7833 | 0.7090 | 2.5549 | 9234.8 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6087 | 0.6783 | 0.6172 | 0.1973 | 9303.9 | 0/23 |
| Illumination | 0.6182 | 0.6909 | 0.6309 | 0.1978 | 9335.4 | 0/11 |
| Viewpoint | 0.6000 | 0.6667 | 0.6046 | 0.1968 | 9275.0 | 0/12 |

#### D = 25000

Usable MIRFlickr distractors: 24985 (15 skipped without SIFT descriptors), database size: 25100

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6870 | 0.7391 | 0.6866 | 6.2065 | 25100.0 | 0/23 |
| Illumination | 0.6909 | 0.7091 | 0.6768 | 6.1750 | 25100.0 | 0/11 |
| Viewpoint | 0.6833 | 0.7667 | 0.6956 | 6.2354 | 25100.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6435 | 0.6870 | 0.6540 | 3.6440 | 8987.0 | 0/23 |
| Illumination | 0.6545 | 0.6727 | 0.6424 | 3.4356 | 8349.6 | 0/11 |
| Viewpoint | 0.6333 | 0.7000 | 0.6646 | 3.8351 | 9571.2 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.2435 | 0.2435 | 0.2348 | 1.1145 | 1979.7 | 0/23 |
| Illumination | 0.3636 | 0.3636 | 0.3455 | 1.0974 | 1957.5 | 0/11 |
| Viewpoint | 0.1333 | 0.1333 | 0.1333 | 1.1301 | 2000.0 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6870 | 0.7391 | 0.6866 | 8.2553 | 24803.0 | 0/23 |
| Illumination | 0.6909 | 0.7091 | 0.6768 | 8.2368 | 24803.0 | 0/11 |
| Viewpoint | 0.6833 | 0.7667 | 0.6956 | 8.2723 | 24803.0 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.5565 | 0.6087 | 0.5811 | 0.3851 | 24830.0 | 0/23 |
| Illumination | 0.6000 | 0.6364 | 0.6160 | 0.3845 | 24830.0 | 0/11 |
| Viewpoint | 0.5167 | 0.5833 | 0.5491 | 0.3856 | 24830.0 | 0/12 |

## K = 1024

### Selected validation parameters

| Method | Parameters | Validation mAP@10 | Mean candidates |
|---|---|---:|---:|
| LSH | k=4, L=9, w=0.1 | 0.7054 | 4139.9 |
| HYPERCUBE | kproj=10, M=2000, w=0.05, probes=40 | 0.3870 | 1299.7 |
| IVFFLAT | kclusters=8, nprobe=2 | 0.7404 | 8701.8 |
| IVFPQ | kclusters=6, nprobe=2, M=16, nbits=8 | 0.5879 | 8713.5 |

### Validation results

Database: 10115 vectors (10000 MIRFlickr distractors).

#### EXACT (reference)

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| - | 0.7391 | 0.7913 | 0.7436 | 4.8529 | 10115.0 | 0/23 |  |

#### LSH

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| k=4, L=3, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0099 | 0.7 | 23/23 |  |
| k=4, L=5, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0165 | 1.1 | 23/23 |  |
| k=4, L=7, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0224 | 1.3 | 23/23 |  |
| k=4, L=9, w=0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0303 | 1.7 | 23/23 |  |
| k=4, L=3, w=0.025 | 0.0522 | 0.0522 | 0.0522 | 0.0233 | 20.6 | 9/23 |  |
| k=4, L=5, w=0.025 | 0.0957 | 0.0957 | 0.0957 | 0.0448 | 34.2 | 4/23 |  |
| k=4, L=7, w=0.025 | 0.1043 | 0.1043 | 0.1043 | 0.0648 | 48.2 | 3/23 |  |
| k=4, L=9, w=0.025 | 0.1304 | 0.1304 | 0.1304 | 0.0842 | 61.5 | 1/23 |  |
| k=4, L=3, w=0.05 | 0.2957 | 0.3043 | 0.2978 | 0.2117 | 258.7 | 1/23 |  |
| k=4, L=5, w=0.05 | 0.3826 | 0.3826 | 0.3754 | 0.3593 | 439.9 | 0/23 |  |
| k=4, L=7, w=0.05 | 0.4609 | 0.4609 | 0.4464 | 0.4723 | 589.3 | 0/23 |  |
| k=4, L=9, w=0.05 | 0.4957 | 0.5043 | 0.4832 | 0.6186 | 730.4 | 0/23 |  |
| k=4, L=3, w=0.1 | 0.5304 | 0.5478 | 0.5260 | 1.2867 | 1887.0 | 0/23 |  |
| k=4, L=5, w=0.1 | 0.6174 | 0.6522 | 0.6180 | 1.8216 | 2782.9 | 0/23 |  |
| k=4, L=7, w=0.1 | 0.6783 | 0.7130 | 0.6722 | 2.2808 | 3569.2 | 0/23 |  |
| k=4, L=9, w=0.1 | 0.7130 | 0.7391 | 0.7054 | 2.6672 | 4139.9 | 0/23 | yes |

#### HYPERCUBE

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| kproj=10, M=250, w=0.05, probes=5 | 0.1130 | 0.1130 | 0.1130 | 0.1343 | 157.8 | 0/23 |  |
| kproj=10, M=250, w=0.05, probes=10 | 0.1652 | 0.1652 | 0.1609 | 0.1827 | 217.2 | 0/23 |  |
| kproj=10, M=250, w=0.05, probes=20 | 0.1652 | 0.1652 | 0.1609 | 0.1966 | 233.3 | 0/23 |  |
| kproj=10, M=250, w=0.05, probes=40 | 0.1826 | 0.1826 | 0.1783 | 0.2192 | 250.0 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=5 | 0.1217 | 0.1217 | 0.1217 | 0.1953 | 233.3 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=10 | 0.2087 | 0.2087 | 0.2058 | 0.3044 | 366.3 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=20 | 0.2174 | 0.2174 | 0.2145 | 0.3529 | 421.1 | 0/23 |  |
| kproj=10, M=500, w=0.05, probes=40 | 0.2348 | 0.2348 | 0.2319 | 0.3911 | 470.3 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=5 | 0.1217 | 0.1217 | 0.1217 | 0.1981 | 233.3 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=10 | 0.2261 | 0.2261 | 0.2232 | 0.4477 | 539.5 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=20 | 0.2870 | 0.2870 | 0.2841 | 0.6136 | 701.4 | 0/23 |  |
| kproj=10, M=1000, w=0.05, probes=40 | 0.3130 | 0.3130 | 0.3087 | 0.7019 | 845.0 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=5 | 0.1217 | 0.1217 | 0.1217 | 0.1973 | 233.3 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=10 | 0.2435 | 0.2435 | 0.2406 | 0.6501 | 571.0 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=20 | 0.3478 | 0.3478 | 0.3449 | 0.8227 | 994.5 | 0/23 |  |
| kproj=10, M=2000, w=0.05, probes=40 | 0.3913 | 0.3913 | 0.3870 | 1.0877 | 1299.7 | 0/23 | yes |

#### IVFFLAT

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| kclusters=5, nprobe=1 | 0.6870 | 0.7304 | 0.6930 | 2.9844 | 5430.1 | 0/23 |  |
| kclusters=5, nprobe=2 | 0.7391 | 0.7826 | 0.7401 | 5.0045 | 8735.4 | 0/23 |  |
| kclusters=5, nprobe=3 | 0.7391 | 0.7913 | 0.7436 | 5.7155 | 9891.0 | 0/23 |  |
| kclusters=5, nprobe=5 | 0.7391 | 0.7913 | 0.7436 | 5.8263 | 10115.0 | 0/23 |  |
| kclusters=6, nprobe=1 | 0.6696 | 0.7043 | 0.6714 | 2.9562 | 5568.8 | 0/23 |  |
| kclusters=6, nprobe=2 | 0.7391 | 0.7826 | 0.7404 | 4.9069 | 8713.5 | 0/23 |  |
| kclusters=6, nprobe=3 | 0.7391 | 0.7913 | 0.7436 | 5.6655 | 9884.0 | 0/23 |  |
| kclusters=6, nprobe=5 | 0.7391 | 0.7913 | 0.7436 | 5.9150 | 10109.3 | 0/23 |  |
| kclusters=8, nprobe=1 | 0.6696 | 0.7043 | 0.6697 | 3.0854 | 5568.1 | 0/23 |  |
| kclusters=8, nprobe=2 | 0.7391 | 0.7826 | 0.7404 | 4.9588 | 8701.8 | 0/23 | yes |
| kclusters=8, nprobe=3 | 0.7391 | 0.7913 | 0.7436 | 5.7401 | 9872.0 | 0/23 |  |
| kclusters=8, nprobe=5 | 0.7391 | 0.7913 | 0.7436 | 5.9352 | 10094.2 | 0/23 |  |
| kclusters=10, nprobe=1 | 0.7304 | 0.7739 | 0.7320 | 4.6926 | 9387.7 | 0/23 |  |
| kclusters=10, nprobe=2 | 0.7391 | 0.7913 | 0.7436 | 5.0107 | 9889.6 | 0/23 |  |
| kclusters=10, nprobe=3 | 0.7391 | 0.7913 | 0.7436 | 4.9882 | 9942.3 | 0/23 |  |
| kclusters=10, nprobe=5 | 0.7391 | 0.7913 | 0.7436 | 4.9978 | 9997.5 | 0/23 |  |

#### IVFPQ

| Parameters | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| kclusters=5, nprobe=1, M=16, nbits=8 | 0.5478 | 0.5826 | 0.5451 | 0.1437 | 5430.1 | 0/23 |  |
| kclusters=5, nprobe=2, M=16, nbits=8 | 0.5826 | 0.6261 | 0.5792 | 0.2523 | 8735.4 | 0/23 |  |
| kclusters=5, nprobe=3, M=16, nbits=8 | 0.5826 | 0.6261 | 0.5792 | 0.3359 | 9891.0 | 0/23 |  |
| kclusters=5, nprobe=5, M=16, nbits=8 | 0.5826 | 0.6261 | 0.5792 | 0.4731 | 10115.0 | 0/23 |  |
| kclusters=6, nprobe=1, M=16, nbits=8 | 0.5130 | 0.5739 | 0.5294 | 0.1434 | 5568.8 | 0/23 |  |
| kclusters=6, nprobe=2, M=16, nbits=8 | 0.5652 | 0.6348 | 0.5879 | 0.2524 | 8713.5 | 0/23 | yes |
| kclusters=6, nprobe=3, M=16, nbits=8 | 0.5652 | 0.6348 | 0.5879 | 0.3366 | 9884.0 | 0/23 |  |
| kclusters=6, nprobe=5, M=16, nbits=8 | 0.5652 | 0.6348 | 0.5879 | 0.4792 | 10109.3 | 0/23 |  |
| kclusters=8, nprobe=1, M=16, nbits=8 | 0.5043 | 0.5739 | 0.4933 | 0.1460 | 5568.1 | 0/23 |  |
| kclusters=8, nprobe=2, M=16, nbits=8 | 0.5565 | 0.6435 | 0.5527 | 0.2561 | 8701.8 | 0/23 |  |
| kclusters=8, nprobe=3, M=16, nbits=8 | 0.5565 | 0.6435 | 0.5527 | 0.4687 | 9872.0 | 0/23 |  |
| kclusters=8, nprobe=5, M=16, nbits=8 | 0.5565 | 0.6435 | 0.5527 | 0.5216 | 10094.2 | 0/23 |  |
| kclusters=10, nprobe=1, M=16, nbits=8 | 0.5130 | 0.6087 | 0.5327 | 0.2057 | 9387.7 | 0/23 |  |
| kclusters=10, nprobe=2, M=16, nbits=8 | 0.5217 | 0.6174 | 0.5414 | 0.2801 | 9889.6 | 0/23 |  |
| kclusters=10, nprobe=3, M=16, nbits=8 | 0.5217 | 0.6174 | 0.5414 | 0.3486 | 9942.3 | 0/23 |  |
| kclusters=10, nprobe=5, M=16, nbits=8 | 0.5217 | 0.6174 | 0.5414 | 0.4890 | 9997.5 | 0/23 |  |

### Test results

#### D = 1000

Usable MIRFlickr distractors: 1000 (0 skipped without SIFT descriptors), database size: 1115

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6696 | 0.7478 | 0.6974 | 0.5535 | 1115.0 | 0/23 |
| Illumination | 0.6545 | 0.6727 | 0.6577 | 0.5563 | 1115.0 | 0/11 |
| Viewpoint | 0.6833 | 0.8167 | 0.7338 | 0.5510 | 1115.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6522 | 0.6957 | 0.6605 | 0.3012 | 470.1 | 0/23 |
| Illumination | 0.6000 | 0.6182 | 0.6032 | 0.3042 | 481.3 | 0/11 |
| Viewpoint | 0.7000 | 0.7667 | 0.7131 | 0.2985 | 459.8 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.3130 | 0.3391 | 0.3094 | 0.0700 | 120.9 | 0/23 |
| Illumination | 0.2545 | 0.2727 | 0.2247 | 0.0698 | 118.9 | 0/11 |
| Viewpoint | 0.3667 | 0.4000 | 0.3870 | 0.0702 | 122.8 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6696 | 0.7565 | 0.7000 | 0.5689 | 1070.5 | 0/23 |
| Illumination | 0.6545 | 0.6727 | 0.6577 | 0.5708 | 1089.0 | 0/11 |
| Viewpoint | 0.6833 | 0.8333 | 0.7388 | 0.5672 | 1053.6 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7043 | 0.7826 | 0.7178 | 0.1535 | 1093.0 | 0/23 |
| Illumination | 0.7273 | 0.7455 | 0.7222 | 0.1545 | 1093.0 | 0/11 |
| Viewpoint | 0.6833 | 0.8167 | 0.7137 | 0.1526 | 1093.0 | 0/12 |

#### D = 5000

Usable MIRFlickr distractors: 5000 (6 skipped without SIFT descriptors), database size: 5115

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6696 | 0.7304 | 0.6817 | 2.5261 | 5115.0 | 0/23 |
| Illumination | 0.6545 | 0.6727 | 0.6517 | 2.4176 | 5115.0 | 0/11 |
| Viewpoint | 0.6833 | 0.7833 | 0.7093 | 2.6256 | 5115.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6435 | 0.6957 | 0.6489 | 1.3587 | 2120.8 | 0/23 |
| Illumination | 0.6000 | 0.6182 | 0.5941 | 1.3876 | 2164.6 | 0/11 |
| Viewpoint | 0.6833 | 0.7667 | 0.6992 | 1.3322 | 2080.6 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.2870 | 0.3391 | 0.3045 | 0.4627 | 556.0 | 0/23 |
| Illumination | 0.2000 | 0.2727 | 0.2144 | 0.4657 | 551.0 | 0/11 |
| Viewpoint | 0.3667 | 0.4000 | 0.3870 | 0.4599 | 560.6 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6696 | 0.7304 | 0.6817 | 2.5045 | 5012.0 | 0/23 |
| Illumination | 0.6545 | 0.6727 | 0.6517 | 2.5127 | 5012.0 | 0/11 |
| Viewpoint | 0.6833 | 0.7833 | 0.7093 | 2.4971 | 5012.0 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.5913 | 0.7043 | 0.6151 | 0.2095 | 5048.0 | 0/23 |
| Illumination | 0.6182 | 0.7091 | 0.6310 | 0.2102 | 5048.0 | 0/11 |
| Viewpoint | 0.5667 | 0.7000 | 0.6006 | 0.2089 | 5048.0 | 0/12 |

#### D = 10000

Usable MIRFlickr distractors: 10000 (7 skipped without SIFT descriptors), database size: 10115

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6609 | 0.7217 | 0.6783 | 4.8689 | 10115.0 | 0/23 |
| Illumination | 0.6545 | 0.6727 | 0.6517 | 4.8715 | 10115.0 | 0/11 |
| Viewpoint | 0.6667 | 0.7667 | 0.7026 | 4.8666 | 10115.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6435 | 0.6957 | 0.6485 | 2.6169 | 4129.8 | 0/23 |
| Illumination | 0.6000 | 0.6182 | 0.5941 | 2.6713 | 4230.4 | 0/11 |
| Viewpoint | 0.6833 | 0.7667 | 0.6983 | 2.5670 | 4037.6 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.2870 | 0.3217 | 0.3001 | 0.8411 | 1004.0 | 0/23 |
| Illumination | 0.2000 | 0.2364 | 0.2053 | 0.8285 | 984.0 | 0/11 |
| Viewpoint | 0.3667 | 0.4000 | 0.3870 | 0.8526 | 1022.3 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6609 | 0.7217 | 0.6783 | 5.0397 | 9936.3 | 0/23 |
| Illumination | 0.6545 | 0.6727 | 0.6517 | 5.0361 | 9947.2 | 0/11 |
| Viewpoint | 0.6667 | 0.7667 | 0.7026 | 5.0430 | 9926.2 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.5826 | 0.6435 | 0.5836 | 0.2626 | 9928.3 | 0/23 |
| Illumination | 0.6000 | 0.6727 | 0.5948 | 0.2619 | 9939.2 | 0/11 |
| Viewpoint | 0.5667 | 0.6167 | 0.5732 | 0.2633 | 9918.2 | 0/12 |

#### D = 25000

Usable MIRFlickr distractors: 24985 (15 skipped without SIFT descriptors), database size: 25100

##### EXACT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6522 | 0.7217 | 0.6754 | 12.2440 | 25100.0 | 0/23 |
| Illumination | 0.6364 | 0.6727 | 0.6456 | 12.2991 | 25100.0 | 0/11 |
| Viewpoint | 0.6667 | 0.7667 | 0.7026 | 12.1935 | 25100.0 | 0/12 |

##### LSH

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6435 | 0.6870 | 0.6449 | 6.5863 | 10340.5 | 0/23 |
| Illumination | 0.6000 | 0.6182 | 0.5886 | 6.7704 | 10619.7 | 0/11 |
| Viewpoint | 0.6833 | 0.7500 | 0.6964 | 6.4175 | 10084.6 | 0/12 |

##### HYPERCUBE

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.2174 | 0.2174 | 0.2130 | 1.4292 | 1649.2 | 0/23 |
| Illumination | 0.1091 | 0.1091 | 0.1091 | 1.3711 | 1576.9 | 0/11 |
| Viewpoint | 0.3167 | 0.3167 | 0.3083 | 1.4825 | 1715.4 | 0/12 |

##### IVFFLAT

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6522 | 0.7217 | 0.6754 | 13.0966 | 23365.1 | 0/23 |
| Illumination | 0.6364 | 0.6727 | 0.6456 | 12.8180 | 23086.3 | 0/11 |
| Viewpoint | 0.6667 | 0.7667 | 0.7026 | 13.3520 | 23620.8 | 0/12 |

##### IVFPQ

| Category | Recall@5 | Recall@10 | mAP@10 | Mean query time (ms) | Mean candidates | Queries with < 10 results |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.5304 | 0.6000 | 0.5429 | 0.5113 | 23466.6 | 0/23 |
| Illumination | 0.5455 | 0.6000 | 0.5511 | 0.5042 | 23238.4 | 0/11 |
| Viewpoint | 0.5167 | 0.6000 | 0.5353 | 0.5178 | 23675.8 | 0/12 |
