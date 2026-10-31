# Image Search with SIFT and Bag of Visual Words

First programming assignment of *Software Development for Algorithmic Problems*
(K23γ, winter semester 2026–27). Images are turned into fixed-length
Bag-of-Visual-Words (BoVW) vectors using SIFT descriptors and k-means, and five
nearest-neighbour methods are implemented and compared on them: Exact Search,
LSH, Hypercube, IVFFlat and IVFPQ.

The data are the full-size **HPatches** sequences (116 sequences of six images;
`I1` is the query, `I2`–`I6` are its five relevant images) and the
**MIRFlickr-25K** photographs, which are used only as irrelevant distractors.

## Contents

1. [Requirements and build](#1-requirements-and-build)
2. [Usage](#2-usage)
3. [Source files](#3-source-files)
4. [Design and algorithmic decisions](#4-design-and-algorithmic-decisions)
5. [Experimental study](#5-experimental-study)
6. [Pseudocode](#6-pseudocode)

## 1. Requirements and build

- Linux, a C++20 compiler (GCC or Clang), CMake ≥ 3.20, `make`
- OpenCV ≥ 4.4 (SIFT is part of `features2d`); OpenMP is optional

```bash
make                  # Release build in build/, creates the ./search link
make test             # builds and runs the unit tests
make clean
```

The Makefile always builds in `Release` mode, since k-means and BoVW
construction are very slow without optimization. `-Wall -Wextra -Wpedantic` are
enabled.

The unit tests run through CTest and cover the reusable numerical and search
components without requiring the image data sets.

**OpenCV use.** OpenCV is used only for `cv::imread` and
`cv::SIFT::detectAndCompute` (in `extract_sift_descriptors`,
[src/image_features.cpp](src/image_features.cpp)); the descriptors are copied out of the
`cv::Mat` immediately. Descriptor storage, sampling, k-means, nearest-center
assignment and every distance computation are implemented in this project.

**Parallelism.** OpenMP parallelises only the image preprocessing, one image per
iteration: SIFT extraction of the training images and of the HPatches/MIRFlickr/query
images with BoVW construction ([src/pipeline.cpp](src/pipeline.cpp)). Images are
processed in batches of 64 whose results are consumed in input order, so the
output is identical to a serial run and to any thread count. Sampling, k-means,
index construction and **every search run on a single thread**, so query times
are comparable between methods. The thread count is `OMP_NUM_THREADS` if set,
otherwise at most 4. One concurrent SIFT extraction of a large HPatches image
needs about 0.65 GB, so use `OMP_NUM_THREADS=1` or `2` on machines with little RAM.

## 2. Usage

```bash
./search -hp <hpatches_dir> -mir <mirflickr_dir> -split split.txt \
         -vocab <K> -S <S> -D <D> -set <validation|test> -o <output_file> \
         -<method> [method options] -seed <seed>
```

Example:

```bash
./search -hp data/hpatches-sequences-release -mir data/mirflickr \
         -split split.txt -vocab 256 -S 500 -D 10000 \
         -set validation -o output.txt -ivfflat -kclusters 10 -nprobe 2 -seed 1
```

### Common parameters

| Option | Meaning | Default |
|---|---|---|
| `-hp <path>` | HPatches root directory (must exist) | required |
| `-mir <path>` | MIRFlickr-25K directory (must exist) | required |
| `-split <file>` | Training/validation/test split ([split.txt](split.txt)) | required |
| `-vocab <int>` | Number `K` of visual words (k-means centers) | required (see `-experiment`) |
| `-S <int>` | Maximum SIFT descriptors sampled per training image | `500` |
| `-D <int>` | MIRFlickr distractors, one of `1000`, `5000`, `10000`, `25000` | `1000` |
| `-set validation\|test` | Query set | `validation` |
| `-o <file>` | Output file | `output.txt` |
| `-seed <int>` | Seed of every random procedure | `1` |
| `-exact`, `-lsh`, `-hypercube`, `-ivfflat`, `-ivfpq` | Search method; exactly one is required | – |
| `-experiment` | Run the whole study instead of a single method (§5) | – |

### Method parameters

All are optional. Options of another method are rejected.

| Method | Option | Meaning | Default |
|---|---|---|---|
| `-exact` | – | no parameters | – |
| `-lsh` | `-k` | hash functions `h_i` per function `g` | `4` |
| | `-L` | number of hash tables | `5` |
| | `-w` | window width on the projection line | `0.05` |
| `-hypercube` | `-kproj` | hypercube dimension `d'` (at most 20) | `10` |
| | `-w` | window width on the projection line | `0.05` |
| | `-M` | maximum number of candidates checked | `500` |
| | `-probes` | maximum number of vertices probed | `10` |
| `-ivfflat` | `-kclusters` | number of clusters | `max(5, round(n^(1/4)))`, see §4.5 |
| | `-nprobe` | nearest clusters searched | `5` |
| `-ivfpq` | `-kclusters`, `-nprobe` | as for IVFFlat | as above |
| | `-M` | PQ subvectors; must divide `-vocab` | `16` |
| | `-nbits` | bits per code, `2^nbits` centroids per subspace (at most 16) | `8` |

Here `n` is the number of vectors in the database. Invalid input (malformed or
duplicate options, unknown options, non-positive values, `-vocab` not divisible
by `-M`, …) is rejected with an error before any image is processed.

### Output

The file follows the required output format: the method name once, one block
per query, then the means. Values have four decimals, times are in milliseconds.

```text
IVFFLAT
Query: v_bird
Rank-1: v_bird/2
...
Rank-10: im6.jpg
Recall@5: 0.4000
Recall@10: 0.6000
AP@10: 0.4333

Query: i_ski
...

Mean Recall@5: 0.3000
Mean Recall@10: 0.4000
mAP@10: 0.3167
Average query time: 1.0000
```

An approximate method may return fewer than ten candidates. The unfilled
positions are printed as `-` and count as non-relevant in the metrics. The
program prints how many queries had fewer than ten results, because the output
format has no field for it.

### Experiment mode

```bash
./search -hp <hpatches_dir> -mir <mirflickr_dir> -split split.txt \
         -S 500 -seed 1 -o results/experiment.md -experiment
```

Writes the Markdown tables of the experimental study (§5). Without `-vocab` it
compares `K` ∈ {128, 256, 512, 1024}; with `-vocab K` it studies that one size.
Because it chooses `D`, the query sets and all method parameters itself, it
rejects `-D`, `-set`, a method flag and any method option.
[results/experiment.md](results/experiment.md) and
[results/experiment.log](results/experiment.log) are the output and the console log
of the run made with `-S 500 -seed 1`.

## 3. Source files

```text
.
├── Makefile, CMakeLists.txt     build ("make" is the entry point)
├── split.txt                    train/validation/test split of the 116 sequences
├── include/                     public headers, one per module
│   ├── dataset.hpp              ImageRecord, EvaluationDataset, split loading, path construction
│   ├── image_features.hpp       SIFT extraction interface, descriptor sampling
│   ├── matrix.hpp               Matrix<T>, a dense row-major matrix (DescriptorMatrix = Matrix<float>)
│   ├── kmeans.hpp               k-means (k-means++ and Lloyd), used by every clustering step
│   ├── vocabulary.hpp           visual vocabulary and BoVW histogram
│   ├── pipeline.hpp             batch extraction of training samples and BoVW vectors
│   ├── search.hpp               the five search classes, SearchResult, result writer
│   └── experiment.hpp           ExperimentRunner (validation, test, report)
├── src/
│   ├── main.cpp                 argument parsing/validation and the single-run flow
│   ├── dataset.cpp, image_features.cpp, kmeans.cpp, vocabulary.cpp, pipeline.cpp
│   ├── experiment.cpp           experiment mode
│   └── search/
│       ├── common.hpp/.cpp      private helpers shared by the methods (distances, IVF clustering, result building)
│       ├── exact.cpp, lsh.cpp, hypercube.cpp, ivfflat.cpp, ivfpq.cpp
│       └── output.cpp           evaluateAndWriteResults
├── tests/
│   └── unit_tests.cpp           unit tests (k-means, distances, exact search, paths), run by "make test"
└── results/                     output of the final experiment run
```

Headers expose small interfaces and hide the implementation (the search classes
use the pimpl idiom). `main.cpp` only orchestrates and decides the failure
policy; `pipeline.cpp` and `experiment.cpp` provide reusable operations that
can be tested independently of the command line. `src/search/common.hpp` is
not in `include/` because it is internal to the search module. Generated files
(`build/`, `search`, data sets) are ignored by Git.

## 4. Design and algorithmic decisions

### 4.1 Data flow

```text
split.txt ─► training images (70 sequences × 6) ─► SIFT (OpenCV) ─► ≤ S random descriptors per image
                                                                          │
                                                          k-means (K centers) = visual vocabulary
                                                                          │
 HPatches I2–I6 of the chosen split ─┐                                    ▼
 first D usable MIRFlickr images ────┼─► SIFT ─► nearest center per descriptor ─► BoVW vectors (R^K, sum 1)
 HPatches I1 queries ────────────────┘                                    │
                                                    database vectors      │   query vectors
                                                          │               │        │
                                                          ▼               ▼        ▼
                                          index (Exact | LSH | Hypercube | IVFFlat | IVFPQ) ─► top-10 ─► metrics ─► output file
```

### 4.2 Vocabulary and BoVW vectors

- The vocabulary uses only the 70 training sequences, so queries and the
  database never influence it. At most `S` descriptors per image are drawn with
  a seeded `std::mt19937` (all of them if the image has fewer). The union is
  clustered into `K` centers, shared by every image.
- Our k-means ([src/kmeans.cpp](src/kmeans.cpp)) uses k-means++ initialisation and Lloyd
  iterations; an empty cluster is re-seeded with the point farthest from its
  center; distance ties go to the lowest center index. The vocabulary runs at
  most 50 iterations: on the training descriptors this is within 0.1 % of the
  fully converged clustering error, which needs 240–300 iterations.
- A BoVW vector counts, for each center, the image's descriptors nearest to it
  (Euclidean distance) and divides by their total, so entries are non-negative
  and sum to 1. An image with no SIFT descriptors has no vector: it is never
  represented by zeros.

### 4.3 Database and queries

- Queries are `I1` of the sequences of the chosen split (23). The database holds
  `I2`–`I6` of those sequences (115 images) plus `D` MIRFlickr distractors, so
  its size is 1115, 5115, 10115 or 25115 and each query has exactly five
  relevant images.
- The distractors are the first `D` images in ascending *numeric* order
  (`im2.jpg` before `im10.jpg`). An image without SIFT descriptors is skipped
  and replaced by the next one. MIRFlickr-25K has only 24985 usable images
  (15 have no descriptors), so for `D=25000` all of them are used and the actual
  number is reported (database size 25100).
- Failure policy: an unreadable or descriptor-less *training* image is only a
  warning. A failed HPatches database or query image is a fatal error, since a
  query would lose one of its five relevant images or disappear from the evaluation.

### 4.4 Search methods

All methods return the 10 best images of their candidate set. Exact, LSH,
Hypercube and IVFFlat rank candidates by Euclidean distance; IVFPQ ranks by the
asymmetric distance.

- **Exact:** computes the distance to every database vector.
- **LSH:** `L` tables; each combines `k` hash functions
  `h_i(p) = ⌊(p·v_i + t_i) / w⌋` (Gaussian `v_i`, `t_i` uniform in `[0, w)`) as
  `g(p) = (Σ r_i h_i(p)) mod 1000000007` with seeded random `r_i`. The candidates
  are the points in the query's bucket of any table, duplicates removed.
- **Hypercube:** `d'` projections `h_i` as above; a seeded integer hash maps each
  window number to a bit, so a point becomes a `d'`-bit vertex. A query visits
  the first `probes` vertices in increasing Hamming distance from its own vertex
  (ties in ascending vertex number, generated directly rather than by sorting
  all `2^d'`) and stops after `M` candidates.
- **IVFFlat:** `kclusters` coarse centers from k-means; every database vector is
  appended to the inverted list of its nearest center. A query takes the `nprobe`
  nearest centers and ranks the vectors of their lists by exact distance.
- **IVFPQ:** same coarse structure, but each vector is stored as the product-quantized
  *residual* (vector minus its coarse center): the `d`-dimensional residual is split
  into `M` subvectors, and each subspace has a k-means codebook of `2^nbits`
  centroids. A query builds, for each probed list, an `M × 2^nbits` lookup table
  of squared distances from its residual to the centroids; the ADC distance of a
  stored vector is the sum of `M` table entries. The 10 smallest are returned,
  **without reranking by exact distance**. We train the codebooks on the residuals of all `n` database
  vectors (our choice). With `nbits=8` the common guideline of about 39 training
  points per centroid is met from `n ≈ 10000`; at `D=1000` each centroid gets
  only about 4, which affects IVFPQ accuracy there.

### 4.5 IVF clustering

For IVFFlat and IVFPQ the coarse k-means runs on a **random sample of about √n**
of the `n` database vectors, not on all of them. The number of centers is
`-kclusters` and is independent of the sample size. Choosing `kclusters = ⌈√n⌉`
is not appropriate here, so a smaller number of clusters is selected from the
validation experiments. Afterwards **all** `n` vectors (sampled ones included) are
assigned to their nearest center and inserted into the inverted lists, so no
image is excluded from the search. We therefore:

- train on a seeded random sample of ⌈√n⌉ vectors (`trainIVFClusters`,
  [src/search/common.cpp](src/search/common.cpp)) and assign all `n` vectors afterwards;
- use `max(5, round(n^(1/4)))` as the default `kclusters` (6, 8, 10 and 13 for
  the four database sizes) — this formula is our own choice; the default `nprobe` is 5;
- lower a `-kclusters` larger than the sample size to ⌈√n⌉ with a warning, since
  k-means cannot form more clusters than it has points (the line
  `resolved IVF parameters` shows the values actually used).

The defaults are resolved after the database is built, so `n` is its actual size.

### 4.6 Evaluation and timing

For a query `Q` with relevant set `R(Q)` (five images) and ranking `Top_n(Q)`:
`Recall@n = |Top_n ∩ R| / 5`, and `AP@10 = (1/5) Σ_j j / r_j` over the ranks
`r_1 < … < r_m` of the relevant images found in the top 10; `mAP@10` is the mean
over queries. The query time covers one `search` call, from the available BoVW
vector to the returned ranking (index construction, image reading, SIFT and the
metric computation are excluded). A single run searches each query once and
reports that time. Experiment mode is described in §5.

### 4.7 Reproducibility

Every random choice (descriptor sampling, k-means, LSH/Hypercube projections,
IVF sampling, PQ codebooks) derives from `-seed` (default 1), so a run with the
same seed and data gives the same rankings and metrics, independent of the number
of threads.

## 5. Experimental study

`./search ... -experiment` (§2) performs the whole experimental study:

1. **Preprocessing once:** training descriptors are sampled once and every `K` is
   trained on the same sample; each validation, test and MIRFlickr image is
   passed through SIFT once and encoded with every vocabulary; the first 25000
   usable MIRFlickr images are encoded once, and every `D` uses a prefix of them.
2. **Tuning on validation** (23 queries, `D=10000`): two parameters per method are
   varied over 16 combinations, the others are fixed.

   | Method | Varied | Fixed |
   |---|---|---|
   | LSH | `w` ∈ {0.01, 0.025, 0.05, 0.1}, `L` ∈ {3, 5, 7, 9} | `k=4` |
   | Hypercube | `M` ∈ {250, 500, 1000, 2000}, `probes` ∈ {5, 10, 20, 40} | `kproj=10`, `w=0.05` |
   | IVFFlat | `kclusters` ∈ {5, 6, 8, 10}, `nprobe` ∈ {1, 2, 3, 5} | – |
   | IVFPQ | `kclusters` ∈ {5, 6, 8, 10}, `nprobe` ∈ {1, 2, 3, 5} | `M=16`, `nbits=8` |

   The `kclusters` values are our own choice of small cluster counts (§4.5), up
   to about `n^(1/4) ≈ 10` for the validation database of 10115 vectors.
3. **Selection rule:** configurations whose validation mAP@10 is within 0.01 of
   the method's best count as equally accurate (with 23 queries one query's AP
   moving by 0.23 changes mAP@10 by 0.01); among them the one with the fewest mean
   candidates (distinct distances computed per query) is chosen, the first in grid
   order on ties. The candidate count is deterministic, unlike time, so the same
   seed always selects the same parameters. `K` is selected the same way from the
   exact-search validation mAP@10 (smallest `K` within 0.01 of the best).
4. **Test:** the selected parameters of each method, `kclusters` included, are used
   unchanged on the test queries for `D` ∈ {1000, 5000, 10000, 25000}, reporting
   Recall@5, Recall@10, mAP@10, mean query time, mean candidates and the number of
   queries with fewer than ten results — overall and for the illumination and
   viewpoint sequences separately.
5. **Timing:** per configuration, the query set first runs once untimed (warm-up);
   then 5 timed passes run over the whole set. A query's time is the median of
   its 5 runs and the reported time is the mean of these medians. Whole passes are
   repeated, rather than one query several times in a row, so a query does not find
   its own candidates still cached.

The result tables are in [results/experiment.md](results/experiment.md).

## 6. Pseudocode

```text
main(args):
    validate args                                   # all checks before any image is read
    if -experiment: return ExperimentRunner(args).run(output)

    splits       <- load split.txt                  # 70 train / 23 validation / 23 test
    descriptors  <- for each training image (parallel): SIFT, keep <= S random descriptors
    vocabulary   <- kmeans(union of descriptors, K)
    dataset      <- queries I1, database I2..I6, MIRFlickr pool sorted numerically
    database     <- BoVW(HPatches database images)               # any failure is fatal
    for image in pool until D usable images are collected:       # skip images without SIFT
        database.append(BoVW(image))
    resolve IVF defaults from n = |database|
    queries      <- BoVW(query images)                           # any failure is fatal
    index        <- build Exact | LSH | Hypercube | IVFFlat | IVFPQ on database
    for q in queries:
        start timer
        ranking <- index.search(q)                  # top 10 of the candidate set
        stop timer
        recall@5, recall@10, AP@10 <- compare ranking with the 5 relevant images of q
    write rankings, metrics and their means to the output file

ExperimentRunner.run(output):
    sample training descriptors once; train a vocabulary for each K
    encode validation, test and the first 25000 usable MIRFlickr images for every K
    for K in vocabulary sizes:
        for method in LSH, Hypercube, IVFFlat, IVFPQ:
            run all 16 configurations on validation (D = 10000), selection rule of §5
        for D in 1000, 5000, 10000, 25000:
            run Exact and the selected configurations on test; aggregate overall / illumination / viewpoint
    select K from exact-search validation mAP@10; write the Markdown report
```
