# Geometric Verification and Reranking with SIFT and RANSAC

Second programming assignment of *Software Development for Algorithmic
Problems*. It extends the first assignment([README.md](README.md)). 
That assignment ranks images by their
Bag-of-Visual-Words vectors, and BoVW ignores where the features are in
the image. Here the first `C` images returned by a search method are checked
geometrically. SIFT descriptors of the query and each candidate are matched,
a homography is estimated with normalized DLT inside RANSAC, and the
candidates are reranked by the number of matches consistent with that
homography.

The data, the split, the vocabulary, the BoVW vectors and the five search
methods are the same as in the first assignment. The new program is `verify`;
`search` is unchanged.

## Contents

1. [Requirements and build](#1-requirements-and-build)
2. [Usage](#2-usage)
3. [Source files](#3-source-files)
4. [Design and algorithmic decisions](#4-design-and-algorithmic-decisions)
5. [Experimental study](#5-experimental-study)
6. [Pseudocode](#6-pseudocode)
7. [Use of AI tools](#7-use-of-ai-tools)

## 1. Requirements and build

The requirements are those of the first assignment, plus Eigen:

- Linux, a C++20 compiler, CMake ≥ 3.20, `make`
- OpenCV ≥ 4.4 (`core`, `imgcodecs`, `features2d`); OpenMP is optional
- Eigen ≥ 3.3, header-only (`sudo apt install libeigen3-dev`)

```bash
make         # Release build in build/, creates the ./search and ./verify links
make test    # builds and runs the unit tests (no image data needed)
make clean
```

We developed and ran the project with GCC 11.4, CMake 3.22, OpenCV 4.5.4 and
Eigen 3.4.0.

**Library use.**

- OpenCV is used only to read images (`cv::imread`) and to extract SIFT
  keypoints and descriptors (`cv::SIFT::detectAndCompute`, in
  [src/geometry/local_features.cpp](src/geometry/local_features.cpp)).
- Eigen is used only for the SVD: of the DLT matrix
  (`Eigen::JacobiSVD` with `ComputeFullV`), and of the final 3×3 homography
  for the invertibility check, both in
  [src/geometry/homography.cpp](src/geometry/homography.cpp).
- The project's own code does everything else: descriptor matching, the Lowe
  test, the construction of the DLT matrix, normalization and
  denormalization, the degeneracy and numerical checks, RANSAC, the refit,
  reranking and evaluation.
- `cv::findHomography`, `cv::BFMatcher` and similar routines are not used.

**Parallelism.**

- Only SIFT extraction runs in parallel with OpenMP, as in the first
  assignment. This covers the vocabulary and BoVW preprocessing and the
  extraction of local features for verification.
- Candidate search, matching, RANSAC and reranking run on a single thread, so
  the measured times can be compared.

## 2. Usage

`verify` accepts every option of `search` (README.md §2) with the same meaning
and validation, plus six options of its own:

```bash
./verify -hp <hpatches_dir> -mir <mirflickr_dir> -split split.txt \
         -vocab <K> -S <S> -D <D> -set <validation|test> -o <output_file> \
         -<method> [method options] \
         [-C <int>] [-rho <double>] [-tau <double>] [-T <int>] [-p <double>] [-adaptive <0|1>] \
         -seed <seed>
```

Example:

```bash
./verify -hp data/hpatches-sequences-release -mir data/mirflickr \
         -split split.txt -vocab 256 -S 500 -D 10000 \
         -set validation -ivfflat -kclusters 16 -nprobe 5 \
         -C 50 -rho 0.8 -tau 3 -T 2000 -p 0.99 -adaptive 1 \
         -seed 1 -o results.txt
```

| Option | Meaning | Allowed values | Default |
|---|---|---|---|
| `-C <int>` | number of candidate images `TopC(Q)` verified per query | ≥ 10 | `50` |
| `-rho <double>` | Lowe ratio: a match is kept when `d1 < ρ·d2` | `0 < ρ < 1` | `0.8` |
| `-tau <double>` | inlier threshold in pixels: `e_i(H) < τ` | finite, `> 0` | `3` |
| `-T <int>` | maximum number of RANSAC trials `Tmax` | `> 0` | `2000` |
| `-p <double>` | confidence of the adaptive trial limit | `0 < p < 1` | `0.99` |
| `-adaptive <0\|1>` | lower the trial limit as better hypotheses are found | `0`, `1` | `1` |

Differences from `search`:

- `-kclusters` is **required** for `-ivfflat` and `-ivfpq`, since `verify` has
  no default cluster count. A value above the IVF training sample size is
  still lowered to it with a warning (README.md §4.5).
- `-experiment` is rejected: each `verify` call is a single run.
- `-seed` also seeds RANSAC.
- `C ≥ 10` because the output always shows ten ranks before and after
  reranking.
- Every check runs before any image is read. If the output file cannot be
  opened, `verify` stops before preprocessing starts.

At the end of a run `verify` prints three lines: the output path, how many
queries got fewer than `C` candidates, and how many database images had
their SIFT features extracted for verification.

### Output

The file starts with the method name, then has one block per query and the
summary. Values have four decimals and times are in milliseconds. The lines
after `After-AP@10` in each query block, and after `MeanCornerErrorPx` in the
summary, are **additional** fields. They appear after the required ones, so
the required order is unchanged.

```text
IVFFLAT
Query: v_bird
Candidates: 50
Before-Rank-1: v_bird/2
...
Before-Rank-10: im4021.jpg
After-Rank-1: v_bird/3
After-Inliers-1: 812
After-MedianError-1: 0.6123
...
After-Rank-10: im77.jpg
After-Inliers-10: 0
After-MedianError-10: NA
Before-Recall@5: 0.4000
Before-Recall@10: 0.6000
Before-AP@10: 0.4333
After-Recall@5: 0.8000
After-Recall@10: 0.8000
After-AP@10: 0.8000
CandidateTimeMs: 1.2000            <- additional: times of this query
VerificationTimeMs: 11893.4000
TotalTimeMs: 11894.6000
MatchCount-Recall@5: 0.8000        <- additional: ranking by Lowe matches only
MatchCount-Recall@10: 0.8000
MatchCount-AP@10: 0.7600

Query: i_ski
...

MeanBeforeRecall@5: ...
...
AverageCandidateTimeMs: ...
AverageVerificationTimeMs: ...
AverageTotalTimeMs: ...
ValidHomographyRate: ...
AccurateHomographyRate5px: ...
MeanCornerErrorPx: ...
AccurateHomographyRate1px: ...     <- additional
AccurateHomographyRate3px: ...
MeanMatchCountRecall@5: ...
MeanMatchCountRecall@10: ...
MeanMatchCountMAP@10: ...
QueriesBelowC: 0
Illumination-Queries: 11           <- the summary again for the i_ sequences
Illumination-MeanBeforeRecall@5: ...
...
Illumination-QueriesBelowC: 0
Viewpoint-Queries: 12              <- and for the v_ sequences
...
Viewpoint-QueriesBelowC: 0
```

The values are illustrative. The arrows are annotations of this README and
are not part of the file.

- A rank that a short `TopC(Q)` does not fill is printed as `-`, together with
  its inliers and median error. Recall and AP still divide by 5.
- `After-Inliers` is 0 and `After-MedianError` is `NA` for a candidate without
  a valid homography. `MeanCornerErrorPx` is `NA` when no relevant pair has a
  valid estimate. An error is never printed as 0 in place of a missing value.
- `QueriesBelowC` counts the queries whose method returned fewer than `C`
  candidates. Exact search always returns `C`.
- The `Illumination-` and `Viewpoint-` blocks repeat every summary line for
  the queries whose sequence starts with `i_` or `v_`. An empty group is left
  out.

## 3. Source files

The full tree is in README.md §3. The second assignment added these modules:

| File (header in `include/`, source in `src/`) | Contents |
|---|---|
| `geometry/local_features` | SIFT keypoint coordinates with their descriptors in the same row order, plus the image size |
| `geometry/descriptor_matching` | two nearest neighbours per query descriptor and the Lowe test |
| `geometry/homography` | normalized DLT, degeneracy checks of the minimal sample, invertibility check, checked projection |
| `geometry/geometric_verification` | RANSAC with refit, verification of one candidate and of `TopC(Q)`, reranking, match-count ranking |
| `evaluation/geometric_evaluation` | HPatches ground truth, corner error, evaluation of the relevant pairs and of a query, summaries |
| `evaluation/verification_output` | the output file of `verify` |
| `data/pipeline` (`LocalFeatureStore`) | extracts the local features of each image once, in parallel, outside the timers |
| `app/command_line` (`parse_verify_args`) | the options of `verify` on top of the `search` parser |
| `app/search_setup` | vocabulary, BoVW database and queries, shared by `search` and `verify` |
| `search/*` (`candidates`) | the first `C` candidates of each method with their original rank |
| `src/verify_main.cpp` | the `verify` program: per query `TopC`, verification, evaluation |

Geometric verification never sees the HPatches ground truth. The ground truth
is read only in `evaluation/`, after both timed stages of a query.

## 4. Design and algorithmic decisions

### 4.1 Data flow

```text
 vocabulary, BoVW database and queries, index   (exactly as in search, README.md §4.1)
                         │
     query Q = I1        │ candidates(Q, C)                                ── T_candidates
         │               ▼
         │            TopC(Q) ─► SIFT of candidates not yet extracted (stored for later queries)
         ▼               │
 SIFT keypoints of Q     ▼
         └─────► for each I in TopC(Q):                                    ┐
                     match_sift(Q, I, ρ)               ─► M accepted matches
                     estimate_ransac(matches, τ, Tmax, p) ─► H, inliers, median  or failure
                 rerank: inliers ↓, median error ↑, original rank ↑        ┘ ── T_verify
                         │
                         ▼
 first 10 before / after, Recall, AP, corner error with H_1→j^GT  ─►  output file
```

### 4.2 Local features

- `cv::SIFT::create()` with OpenCV's default parameters, so **every**
  keypoint of an image is kept (`nfeatures = 0`). Following the instructor's
  clarification, RANSAC uses all SIFT descriptors of both images. The `-S`
  limit applies only to the descriptors sampled for the vocabulary.
- A point is the keypoint's `pt`, in pixel coordinates of the full-size image
  with `(0,0)` at the center of the top-left pixel. Descriptor row `i`
  belongs to point `i`.
- OpenCV returns SIFT descriptor values as `float`, but every value is a whole
  number between 0 and 255. We store them as **bytes**, which uses 1 byte
  instead of 4 per value and allows exact integer distances (§4.3).
  Extraction checks every value and throws if one cannot be stored exactly,
  so the conversion never loses information. A unit test compares the stored
  bytes with OpenCV's output.
- `LocalFeatureStore` extracts the query features before the first query.
  Candidate features are extracted the first time an image appears in some
  `TopC(Q)`, in parallel, and are kept for later queries. Every image is
  therefore read once, and the extraction is never inside a timer. A database
  image always has descriptors because it has a BoVW vector, so an image
  without descriptors here is a fatal error.

### 4.3 Descriptor matching (`match_descriptors`)

- **Direction Q → I.** For every query descriptor, all descriptors of the
  candidate are scanned. The two smallest squared distances `D1 ≤ D2` and the
  index of the nearest are kept. The resulting homography maps query pixels
  to candidate pixels, the same direction as the ground truth `H_1→j`.
- **Exact integer distances.** The squared distance of two byte descriptors
  is summed in `int32`. The largest possible value is 128·255² = 8,323,200,
  so it cannot overflow and the result is exact. The loop has no early exit,
  so the compiler vectorizes it. Matching takes almost all of `T_verify`
  (§4.10).
- **Lowe test on squared distances.** `d1 < ρ·d2` is the same as
  `D1 < ρ²·D2`, which is compared in `double` with a strict `<`. Only
  accepted matches take a square root, to store their distance.
- **Ties.** Two candidate descriptors at equal distance keep the lower index.
  An exact tie `D1 = D2` fails the strict test, as it should, because the
  match is ambiguous.
- **Edge cases.** A candidate with fewer than two descriptors gives no
  matches. `D2 = 0` (two identical nearest descriptors) is rejected rather
  than divided by.
- **No cross-check.** Several query descriptors may match the same candidate
  keypoint. RANSAC's consensus decides which of these matches fit the
  homography.

### 4.4 Normalized DLT (`estimate_homography`)

1. **Normalization.** For each image separately: translate the centroid of the
   points to the origin and scale by `s = √2 / d̄`, where `d̄` is their mean
   distance from the centroid. This gives `T` for the query points and `T′`
   for the candidate points. A mean distance below `1e-8` px is rejected,
   because `s` would not be reliable.
2. **DLT matrix.** Each correspondence `(x, y) ↔ (u, v)` of normalized points
   adds the two rows `[−x −y −1 0 0 0 ux uy u]` and
   `[0 0 0 −x −y −1 vx vy v]`, so `A` is `2k × 9` for `k` correspondences.
3. **SVD.** `Eigen::JacobiSVD` of `A` itself (not `AᵀA`, which would square
   its condition number) with the **full** `V`. For `k = 4`, `A` is `8 × 9` and
   a thin `V` would lack the ninth column. `h` is the last column of `V`, the
   right singular vector of the smallest singular value. It is reshaped row
   by row into `Hn`.
4. **Denormalization.** `H = T′⁻¹ · Hn · T`.
5. **Validity.** Every element of `H` must be finite. `H` must be numerically
   invertible: its own singular values must satisfy `σ1 > 0` and
   `σ3 / σ1 > 1e-12`. The test does not depend on the arbitrary scale of `H`,
   unlike a threshold on `det H`, which scales with `λ³`.
6. **Projection** (`project_point`). For `H·p̃ = (a, b, c)`, a point is
   projected only if `|c| > 1e-12 · ‖H₃‖ · ‖p̃‖`, where `H₃` is the third row
   of `H`, and if both coordinates are then finite. Otherwise the point is
   treated as projecting to infinity: its reprojection error is `+∞`, so it
   is never an inlier.

All geometry is computed in `double`.

### 4.5 Degeneracy checks

The checks run on every **minimal sample** of four correspondences, in **both**
images, before the DLT:

- **Repeated points:** two points closer than `1e-5` px (squared distance
  below `1e-10`).
- **Collinear triples:** for every triple `a, b, c`,
  `η = |(b − a) × (c − a)| / L²`, where `L` is the longest side. The sample
  is rejected if `L = 0` or `η < 1e-4`. Dividing by `L²` makes `η` independent
  of the image scale.

The refit does not repeat these checks. Its inliers include the
non-degenerate minimal sample, so `A` keeps rank 8. Repeated inlier points,
which are common when several query descriptors match the same candidate
keypoint, only repeat rows of `A`. Rejecting them would discard good
consensus sets.

### 4.6 RANSAC (`estimate_homography_ransac`)

- **Too few matches.** With fewer than four accepted matches, verification
  fails immediately.
- **Random numbers.** Each trial draws four **distinct** matches uniformly
  with `std::mt19937_64`, seeded from `-seed`. The generator is re-seeded for
  every candidate. A pair `(Q, I)` therefore gives the same result whatever
  its position in `TopC(Q)` and whatever `C` or method produced it, so runs
  with different `C` and methods compare the same verification.
- **Rejected trials.** Degenerate samples and invalid homographies are
  skipped but still count as trials.
- **Scoring.** Every hypothesis is scored on all `M` matches. A match is an
  inlier if its reprojection error `‖p′ − π(H·p̃)‖₂` is **strictly** below
  `τ`. A hypothesis replaces the best one if it has more inliers, or the same
  number with a smaller median inlier error. On a further tie the earlier
  trial wins.
- **Adaptive limit** (`-adaptive 1`). The limit starts at `Tmax`. After each
  improvement, with `ŵ = m / M`:

  ```text
  T_limit ← min(T_limit, ⌈log(1−p) / log(1−ŵ⁴)⌉)
  ```

  - Both logarithms are computed with `log1p`, which stays accurate when `ŵ⁴`
    is tiny.
  - `ŵ = 0` leaves the limit unchanged.
  - `ŵ = 1` stops the trials immediately, since every match is an inlier.
  - The limit can only decrease, so it never exceeds `Tmax`.
- **Refit.** With `m ≥ 4` inliers in the best hypothesis, `H` is re-estimated
  by normalized DLT from those `m` correspondences only, with new `T` and
  `T′`. The errors of **all `M`** matches are then recomputed with the refit
  `H`, which gives the final inliers and their median error.
- **Failure.** Verification fails if no valid hypothesis is found, if the
  best one has fewer than four inliers, if the refit is invalid, or if fewer
  than four final inliers remain. A failed candidate has no homography. It is
  never given a placeholder identity matrix.

### 4.7 Reranking

- **Order.** Candidates are sorted by:
  1. number of final inliers `S(I)`, descending;
  2. median inlier error, ascending;
  3. original rank in `TopC(Q)`, ascending.
- **Failed candidates.** A failed candidate has `S(I) = 0` and an infinite
  median error. All failed candidates therefore come after every candidate
  with a valid homography and keep their original relative order.
- **Result.** The first 10 of the new ranking are the result. All `C`
  verified candidates are kept for the evaluation of homography accuracy.

**Match-count ranking (additional).** To compare with a ranking that uses no
geometry, the same candidates are also sorted by the number of matches that
pass the Lowe test, with ties broken by original rank. The matches are
counted during verification, and the sort runs after the timer stops. Its
Recall and AP are the `MatchCount-` lines of the output.

### 4.8 Evaluation

- **Retrieval metrics.**
  - *Before* is the first 10 of `TopC(Q)` in the method's order; *after* is
    the first 10 of the reranked list.
  - Recall@5, Recall@10 and AP@10 use the definitions of the first assignment
    (README.md §4.6) and always divide by the 5 relevant images.
  - Reranking only reorders `TopC(Q)`, so `Recall@10 after ≤ Recall@C before`.
- **Relevant pairs.** For each query the five relevant pairs `(I1, Ij)`,
  `j = 2..6`, are looked up among **all** `C` verified candidates, not only
  the first 10.
  - A pair with a valid homography gets a corner error.
  - A pair outside `TopC(Q)`, or whose verification failed, counts as invalid
    and inaccurate. It stays in the denominator of both rates but gets no
    corner error.
- **Corner error.** The mean, over the four corners `a_i` of `I1`, of
  `‖π(Ĥ·ã_i) − π(H_GT·ã_i)‖₂` in pixels of `Ij`.
  - The corners are the centers of the corner pixels: `(0,0)`, `(w−1,0)`,
    `(w−1,h−1)` and `(0,h−1)`. This is the same coordinate convention as the
    SIFT keypoints that `Ĥ` was fitted to.
  - `H_GT` is read from `<hp>/<sequence>/H_1_j` (9 numbers, row by row), only
    for valid estimates and only after both timers.
  - If an estimated corner does not project to finite coordinates, the
    estimate counts as invalid.
- **Rates.** `ValidHomographyRate` and `AccurateHomographyRateNpx` (strict
  `E_corner < N`, `N` ∈ {1, 3, 5}) divide by all `5·|Q|` relevant pairs, which
  is 115 for a full validation or test set. `MeanCornerErrorPx` averages over
  the valid pairs only.

### 4.9 Timing

- **What is timed.** Both stages are timed with `std::chrono::steady_clock`,
  in milliseconds per query.
  - `T_candidates` is one `candidates(Q, C)` call on the index of the first
    assignment.
  - `T_verify` starts once `TopC(Q)` is available. It covers matching, the
    Lowe test, RANSAC with refit, and reranking of all returned candidates.
  - `T_total = T_candidates + T_verify`.
- **What is not timed:** image reading, SIFT extraction, vocabulary training,
  BoVW construction, index construction and the evaluation.
- **Runs.** Each query runs once.

### 4.10 Reproducibility, running time and memory

- **Reproducibility.** Every random choice derives from `-seed`: vocabulary
  sampling and k-means, the index, and RANSAC. Running again with the same
  options gives the same output file except the time lines.
- **Running time.** Brute-force matching takes about 99% of `T_verify`. An
  HPatches pair has a few thousand descriptors per image, so one candidate
  costs millions of 128-dimensional distances. RANSAC takes a few
  milliseconds per candidate.
- **Our measurements.** On our development machine (validation, `-D 1000`,
  exact search), `T_verify` was about 2.8 s per query for `C = 10` and about
  12 s per query for `C = 50`. A complete validation run with `C = 50`
  therefore needs a few minutes on top of the preprocessing. Storing
  descriptors as bytes with integer distances made `T_verify` about 4–4.5×
  faster than the earlier `float` version, with identical rankings.
- **Memory.** Peak memory (about 4.4 GB) comes from preprocessing, not from
  the stored local features. `OMP_NUM_THREADS=1` lowers it, as in the first
  assignment.

### 4.11 Tests

`make test` runs the unit tests ([tests/unit_tests.cpp](tests/unit_tests.cpp)). Those for the
second assignment cover:

- **Local features:** keypoint/descriptor alignment, the image size, byte
  storage against OpenCV, and the feature store.
- **Matching:** the Lowe test with `ρ` against `ρ²`, ties, the largest
  possible distance, and a brute-force reference.
- **DLT:** identity, translation and projective homographies, repeated and
  collinear points, non-invertible matrices, and projection near infinity.
- **RANSAC:** known homographies with outliers, the exact inlier set, and the
  adaptive limit.
- **Reranking:** tie-breaks, the order of failed candidates, and the
  match-count ranking.
- **Evaluation:** ground-truth parsing, corner error, relevant-pair rates,
  Recall/AP of short rankings, and the output format, including `NA`, `-`,
  the group summaries and `QueriesBelowC`.
- **Command line:** parsing and rejection of every `verify` option.

## 5. Experimental study

The study is submitted as a separate PDF report (up to 5 pages, IEEE
two-column template).

<!-- TODO: file name of the report, how the runs were made (scripts/commands),
     the grid of C ∈ {20, 50, 100}, ρ, τ, Tmax with p = 0.99, the selected
     configuration from validation, and the test runs for D ∈ {1000, 5000,
     10000, 25000}, overall and for illumination/viewpoint. -->

## 6. Pseudocode

The preprocessing (vocabulary, BoVW vectors, index) is the one of `search`,
README.md §6.

```text
verify(args):
    validate args                                    # all checks before any image is read
    vocabulary, database BoVW, query BoVW <- as in search
    index <- build Exact | LSH | Hypercube | IVFFlat | IVFPQ on the database BoVW
    query_features <- SIFT keypoints + descriptors of every query      # parallel, untimed
    store <- empty LocalFeatureStore over the database images
    for each query Q:
        TopC, T_candidates <- index.candidates(BoVW(Q), C)    # original order, maybe < C
        store.load(TopC)                                      # new images only, untimed
        start timer
        for I in TopC:
            matches <- match_sift(Q, I, rho)
            result[I] <- estimate_ransac(matches, tau, Tmax, p, adaptive, seed)
        ranking <- sort TopC by (inliers desc, median asc, original rank asc)
        T_verify <- stop timer
        before, after <- first 10 of TopC, first 10 of ranking
        metrics of before, after and of the match-count ranking
        for j in 2..6: corner error of (I1, Ij) if Ij in TopC with a valid H   # reads H_1_j
    write per-query blocks, summary, Illumination-/Viewpoint- summaries

match_sift(Q, I, rho):
    if |I| < 2: return []
    for each query descriptor q:
        D1, D2, nearest <- two smallest squared distances to I's descriptors (int32)
        if D2 > 0 and D1 < rho^2 * D2: keep (point of q, point of nearest)

estimate_ransac(matches, tau, Tmax, p, adaptive, seed):
    if |matches| < 4: return FAILURE
    rng <- mt19937_64(seed); limit <- Tmax; best <- NONE
    for t in 0 .. limit-1:
        sample <- 4 distinct random matches
        if degenerate(sample): continue                # repeated points, collinear triple
        H <- normalized_DLT(sample); if invalid(H): continue
        inliers <- {i : error(H, matches[i]) < tau}
        if more inliers than best, or equal and smaller median:
            best <- (H, inliers)
            if adaptive: limit <- min(limit, ceil(log1p(-p) / log1p(-w^4))), w = |inliers| / |matches|
    if best = NONE or |best.inliers| < 4: return FAILURE
    H <- normalized_DLT(best.inliers); if invalid(H): return FAILURE
    final <- {i : error(H, matches[i]) < tau}          # all matches again
    if |final| < 4: return FAILURE
    return H, final, median error of final

normalized_DLT(correspondences):
    T, T' <- centroid to origin, mean distance sqrt(2), for each image
    A <- two rows per normalized correspondence
    h <- last column of V in SVD(A) (full V); Hn <- h reshaped row by row
    H <- inverse(T') * Hn * T
    valid if every element finite and sigma3(H) / sigma1(H) > 1e-12
```

## 7. Use of AI tools

<!-- TODO (both students): check every statement in this section, complete the
     parts marked TODO, and describe your own use of AI tools in your own words. -->

We used **Claude Code** (Anthropic) inside VS Code as an assistant. It was
used to review our code against the course material, to explain errors, and
to implement specific, small changes. Each change was proposed in one small
step. We read and tested each one, and committed it ourselves with our own
commit message. The pseudocode of the whole program is in §6.

### Main prompts

Every session started with the same prompt:

> Act as a senior software engineer, with great expertise in Software
> Development for Algorithmic Problems. This is the 2nd assignment of the
> course, and for this particular exercise we have [the assignment
> description], [the lecture] and Φ2.pdf as the exercise's implementation
> tutorial. The description, the lecture and the tutorial must be strictly
> followed so read them carefully. [...] In ASSIGNMENT2_PLAN.md there is a
> thorough recommended implementation plan. [...] it is important to write
> many and small commits, so when we implement we take it step by step. Also,
> use comments of the same format as in the existing code [...]. After every
> implementation, write a small paragraph explaining how the added code
> operates. I will be the one committing, I just need suggested commit
> messages.

It was followed by a task, for example:

- "Review the existing code"
- "Implement the 'The SVD is hand-written' deviation, if it checks with the
  description"
- "Continue with the next immediate step"
- "Does this latest change make the project runnable? [...] Also, is it
  acceptable to the project's guidelines?"
- "yes, go ahead with step 2 (if it's acceptable per the guidelines)"
- "yes, go ahead with step 4. i think we should have a separate README for
  assignment 2"

### Main AI-assisted steps

| Date (2026) | Step | Commits |
|---|---|---|
| 13/11 | Code review against the course material. Fixes: `ρ²` in the squared-distance Lowe test, degeneracy checks only on the minimal sample, Eigen SVD in place of our hand-written Jacobi SVD, `σ3/σ1` invertibility test, unit tests with the course's synthetic examples, image size with the local features | `fcc8bec`–`ade47bb` |
| 14/11 | `H = T′⁻¹ Hn T` with Eigen, reranking, `TopC` for every method, local feature store | `11b6578`–`e3b6d27` |
| 15/11 – 17/11 | Verification of `TopC`, ground truth and corner error, homography rates, before/after metrics, output writer, `verify` command line and program, source folders, match-count ranking | `a48f2ad`–`9a8cd76` |
| 17/11 | Per-query times in the output, byte descriptors with exact integer distances, `QueriesBelowC`, this README | `d919b5d`–`40bc8b4` |

<!-- TODO: the commits of 12/11 – 13/11 before fcc8bec (local features, matching,
     DLT, first RANSAC): say whether and how AI was used for them. -->

### Main AI-assisted debugging steps

- **Lowe test.** The test compared squared distances with `ρ` instead of
  `ρ²`, which accepts far too many matches. Fixed, and a test was added that
  separates the two.
- **Refit.** The degeneracy checks also ran on the refit and rejected good
  consensus sets that contained repeated points. They were moved to the
  minimal sample.
- **Invertibility.** The check used the eighth singular value of `A` and an
  absolute norm threshold on `H`. It was replaced by the scale-invariant ratio
  `σ3/σ1` of `H` itself.
- **Editor error.** VS Code IntelliSense could not find the Eigen headers,
  which are in `/usr/include/eigen3`. The build was not affected, because
  CMake's `Eigen3::Eigen` target supplies the path. The fix was adding that
  path to the editor's include path.
- **Running time.** A first run on real data needed about 48 s of `T_verify`
  per query at `C = 50`, too slow for the experimental study. Profiling
  showed that matching took about 99% of `T_verify`. Byte storage with exact
  integer distances (§4.3) made it about 4× faster without changing any
  ranking.
