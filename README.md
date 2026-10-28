# Image Search with SIFT and Bag of Visual Words

This project is the current implementation of the first algorithmic-problems
assignment. It prepares an image-retrieval dataset using SIFT local descriptors,
learns a visual vocabulary with k-means, and converts images into fixed-size
Bag-of-Visual-Words (BoVW) vectors.

The code prepares the training vocabulary, database vectors, and query vectors,
then searches them with exact search or one of four approximate indexes:
LSH, Hypercube, IVFFlat, and IVFPQ. Exact, LSH, Hypercube, and IVFFlat rank
their candidate set by Euclidean distance; IVFPQ ranks its candidates by the
asymmetric distance (ADC) to their product-quantized codes, as defined in the
lecture. Every method then computes the retrieval metrics and writes the result
file.

## Project structure

```text
.
├── CMakeLists.txt
├── README.md
├── split.txt
├── include/
│   ├── dataset.hpp
│   ├── experiment.hpp
│   ├── image_features.hpp
│   ├── kmeans.hpp
│   ├── matrix.hpp
│   ├── pipeline.hpp
│   ├── search.hpp
│   └── vocabulary.hpp
├── src/
│   ├── experiment.cpp
│   ├── dataset.cpp
│   ├── image_features.cpp
│   ├── kmeans.cpp
│   ├── main.cpp
│   ├── pipeline.cpp
│   ├── vocabulary.cpp
│   └── search/
│       ├── common.hpp
│       ├── common.cpp
│       ├── exact.cpp
│       ├── lsh.cpp
│       ├── hypercube.cpp
│       ├── ivfflat.cpp
│       ├── ivfpq.cpp
│       └── output.cpp
└── .vscode/
    ├── c_cpp_properties.json
    └── settings.json
```

`experiment.cpp` contains the batch validation and test workflow. It is kept
separate from `main.cpp` so the single-run command remains focused on argument
handling and orchestration.

The assignment PDFs and local datasets are external inputs and are not required
to build the executable. Dataset directories are ignored by Git.

## How the modules connect

The executable follows this data flow:

```text
command line
    │
    ▼
src/main.cpp
    │ parses and validates Args
    │
    ├── split.txt ───────────────► dataset module
    │                                  │
    │                                  ├── training image paths
    │                                  └── evaluation records
    │
    ├── pipeline module ◄──────── image paths and dataset records
    │       │
    │       ├── image_features module
    │       │       └── SIFT descriptor matrices (DescriptorMatrix)
    │       │
    │       └── vocabulary module
    │               └── normalized BoVW histograms (K doubles)
    │
    └── vocabulary module ◄────── sampled training descriptors
            └── visual vocabulary (K × 128 DescriptorMatrix)
```

The headers define the public interfaces between modules. The `.cpp` files hide
the implementation details. `main.cpp` is the orchestration layer: it selects
the application policy for failures, while `pipeline.cpp` and
`experiment.cpp` provide reusable image-preparation and batch-evaluation
operations that can be tested independently.

## `CMakeLists.txt`

[`CMakeLists.txt`](CMakeLists.txt) defines the build:

1. Requires CMake 3.20 or newer.
2. Declares a C++ project and requires C++20.
3. Enables `compile_commands.json` generation.
4. Defaults to a `Release` (optimized) build when no build type is given, since
   k-means and BoVW construction are very slow without optimization. Pass
   `-DCMAKE_BUILD_TYPE=Debug` to get a debug build with AddressSanitizer and
   UndefinedBehaviorSanitizer.
5. Locates only the OpenCV components the program uses:
   `find_package(OpenCV REQUIRED COMPONENTS core imgcodecs features2d)`.
6. Builds the `search` executable from all project source files.
7. Adds `include/` to the header search path.
8. Links the executable with those OpenCV libraries.
9. Links OpenMP when the compiler supports it (`find_package(OpenMP)`). It is
   optional: without it the program builds and runs serially with identical
   results.

### Use of OpenCV

OpenCV is used only to read images (`cv::imread`) and to compute SIFT
descriptors (`cv::SIFT::detectAndCompute`), both inside
`extract_sift_descriptors` in `src/image_features.cpp`. The descriptors are
copied out of the `cv::Mat` before that function returns. Everything else,
including descriptor storage, sampling, k-means, nearest-centroid assignment,
and all distance computations, is implemented in this project.

### Parallelism

Image preprocessing runs in parallel with OpenMP;
the search runs serially on a single thread for every method, so query times
are comparable.

Parallel (in [`src/pipeline.cpp`](src/pipeline.cpp), one image per iteration):

- `extract_training_descriptors`: reading each training image and extracting
  its SIFT descriptors.
- `extract_bow_vectors`: reading each HPatches, MIRFlickr, or query image,
  extracting its SIFT descriptors, and building its BoVW histogram.

Serial:

- Sampling the `S` training descriptors per image, in image order with the
  seeded generator.
- k-means for the visual vocabulary and for the IVFFlat/IVFPQ indexes.
- Index construction and every search.

Images are processed in batches of 64 whose results are then consumed in input
order, so the output is identical to a serial run: training samples are drawn
in the same order, and the first `D` usable MIRFlickr images are the same. With
a `D` limit, a batch never exceeds the number of images still needed.

The number of preprocessing threads is `OMP_NUM_THREADS` when that environment
variable is set, and otherwise at most 4. Every concurrent SIFT extraction of a
large HPatches image needs about 0.65 GB, and OpenCV's SIFT already uses
several cores internally for a single image, so more threads mostly add memory.
Measured on a 12-core machine with 14 GB of RAM (`-vocab 64 -D 1000`):

| Threads | Wall time | Peak memory |
|---|---:|---:|
| serial build (no OpenMP) | 142 s | 1.9 GB |
| 2 | 93 s | 2.6 GB |
| 4 (default) | 74 s | 4.5 GB |
| 8 | 66 s | 7.3 GB |
| 12 | 66 s | 8.6 GB |

The results are identical for every thread count. On a machine with less
memory, run with `OMP_NUM_THREADS=2` or `OMP_NUM_THREADS=1`.

The source files are compiled together, so `main.cpp` can include the three
public headers while the linker resolves their implementations from the
corresponding `.cpp` files.

## Dataset module

### `include/dataset.hpp`

[`include/dataset.hpp`](include/dataset.hpp) defines the data structures and
functions used to describe dataset images.

#### `ImageRecord`

```cpp
struct ImageRecord {
    std::string filepath;
    std::string identifier;
};
```

`filepath` is the path used to read the image. `identifier` is the stable name
that will later be written to search results:

- HPatches query: `v_bird`
- HPatches database image: `v_bird/2`
- MIRFlickr image: `im123.jpg`

Keeping these values together prevents the search layer from losing the logical
image name while working with filesystem paths.

#### `EvaluationDataset`

```cpp
struct EvaluationDataset {
    std::vector<ImageRecord> queries;
    std::vector<ImageRecord> database;
    std::vector<ImageRecord> distractor_pool;
};
```

- `queries` contains HPatches image `I1` for every sequence in the selected
  validation or test split.
- `database` contains the related HPatches images `I2` through `I6`.
- `distractor_pool` contains the complete sorted MIRFlickr image list.

The MIRFlickr pool is deliberately kept separate. The application must inspect
descriptors and choose the first `D` usable images, because an image with no
SIFT descriptors must be skipped and replaced by the next image. If the pool
runs out before `D` usable images are found (possible only for `D=25000`, since
MIRFlickr-25K has exactly 25000 images), every usable image is used and the
actual number is reported. Images without
descriptors are never represented by zero vectors.

#### `get_training_image_paths`

```cpp
std::vector<std::filesystem::path> get_training_image_paths(
    const std::string& hp_path,
    const std::vector<std::string>& train_sequences);
```

For every training sequence, this function returns paths for all six HPatches
images (`1.ppm` through `6.ppm`). It does not read images and does not perform
descriptor extraction.

#### `get_evaluation_paths`

```cpp
EvaluationDataset get_evaluation_paths(
    const std::string& hp_dir,
    const std::string& mir_dir,
    const std::string& split_file,
    const std::string& set_type);
```

This function:

1. Reads the split file.
2. Selects either `validation` or `test`.
3. Creates one query record per selected sequence.
4. Creates five HPatches database records per selected sequence.
5. Enumerates MIRFlickr image files.
6. Sorts canonical `im<number>.<extension>` files by their numeric image
   number, so `im2.jpg` comes before `im10.jpg`.
7. Uses a deterministic filename ordering for files that do not match the
   canonical numeric naming pattern.
8. Places every sorted MIRFlickr record in `distractor_pool`.

It returns a complete value instead of mutating vectors supplied by the caller.
Invalid set names, missing split files, malformed lines, invalid subset names,
and duplicate sequence names are reported with exceptions.

### `src/dataset.cpp`

[`src/dataset.cpp`](src/dataset.cpp) implements the dataset API.

The public `load_dataset_splits` function validates each non-empty split-file line as
two fields:

```text
<sequence-name> <train|validation|test>
```

It records all sequence names in a `std::set` to detect duplicates and returns
only the requested subset. The public evaluation function accepts only
`validation` and `test`, because training paths are obtained separately through
`get_training_image_paths`.

The private `list_mirflickr_images` helper scans the MIRFlickr directory and
retains regular `.jpg`, `.jpeg`, `.JPG`, and `.JPEG` files. Canonical filenames
such as `im1.jpg`, `im2.jpg`, and `im10.jpg` are ordered numerically rather than
lexicographically. This is important because the assignment defines the
distractor set as the first `D` images in ascending numerical order. Files
without a valid `im<number>` stem use deterministic filename ordering. The
helper does not inspect image contents; SIFT-based usability filtering is
performed later by `main.cpp`.

## Image-feature module

### `include/image_features.hpp`

[`include/image_features.hpp`](include/image_features.hpp) declares the SIFT
interface.

`SiftExtractionStatus` distinguishes:

- `Success`: descriptors were produced.
- `ImageUnreadable`: OpenCV could not load the image.
- `NoDescriptors`: the image loaded, but SIFT found no descriptors.

`SiftExtractionResult` keeps the descriptor matrix (a `DescriptorMatrix`, see
the matrix module below) and status together so the
caller cannot confuse an unreadable image with an image that simply contains no
keypoints.

### `src/image_features.cpp`

[`src/image_features.cpp`](src/image_features.cpp) implements two operations.

#### `extract_sift_descriptors`

1. Reads the path as grayscale with `cv::imread`.
2. Returns `ImageUnreadable` if the resulting matrix is empty.
3. Creates an OpenCV SIFT detector.
4. Calls `detectAndCompute`.
5. Returns `NoDescriptors` if no descriptor matrix was produced.
6. Otherwise copies the `CV_32F` rows into a `DescriptorMatrix` and returns it
   with `Success`.

Each SIFT row is one 128-dimensional local descriptor.

The pipeline preserves the extraction status for every failed record, so
warnings and errors distinguish an unreadable or missing file from a valid
image for which SIFT produced no descriptors. MIRFlickr scanning skips both
failure types and continues through the numerically ordered files until the
requested number of usable distractors has been collected.

#### `sample_training_descriptors`

The assignment limits the number of descriptors used to train the vocabulary.
This function:

1. Rejects a non-positive maximum.
2. Rejects an empty descriptor matrix.
3. Returns a copy when the image already has at most `S` descriptors.
4. Creates row indices for larger matrices.
5. Shuffles the indices using the caller's seeded `std::mt19937`.
6. Copies the first `S` selected rows into a new matrix.

The random generator is owned by `main.cpp`, so all training-image sampling is
reproducible for a fixed `-seed`.

## Matrix module

[`include/matrix.hpp`](include/matrix.hpp) defines `Matrix<T>`, a dense
row-major matrix with `rows`, `cols`, and a contiguous `std::vector<T> data`.
`row(r)` returns a pointer to the first element of row `r`.
`DescriptorMatrix` is `Matrix<float>` and holds SIFT descriptors or visual
words, one 128-dimensional row each.

## K-means module

[`include/kmeans.hpp`](include/kmeans.hpp) and
[`src/kmeans.cpp`](src/kmeans.cpp) implement the clustering used both for the
visual vocabulary (on `float` SIFT descriptors) and for the IVFFlat/IVFPQ
coarse clusters and PQ codebooks (on `double` BoVW vectors).

```cpp
template <typename T>
KMeansResult<T> kmeans(const Matrix<T>& points, std::size_t clusters, int seed,
                       int max_iterations = 100);
```

1. Rejects empty input, zero clusters, and more clusters than points.
2. Initializes the centers with k-means++: the first center is a uniformly
   random point, and every next center is a point chosen with probability
   proportional to its squared distance from the nearest center chosen so far.
3. Runs Lloyd's algorithm: each iteration moves every center to the mean of
   its points (accumulated in `double`) and then reassigns every point to its
   nearest center.
4. Moves the center of an empty cluster onto the point that is farthest from
   its own center.
5. Stops when no point changes cluster, or after `max_iterations` iterations.
6. Returns the centers, the assignment of every point, the number of
   iterations, and whether it converged.

All random choices use a `std::mt19937` seeded with `seed`, so results are
reproducible. The returned assignments always refer to the returned centers.
`nearest_center` and `squared_distance` are exposed for reuse; distance ties
go to the lowest center index.

## Vocabulary module

### `include/vocabulary.hpp`

[`include/vocabulary.hpp`](include/vocabulary.hpp) exposes:

```cpp
DescriptorMatrix compute_visual_vocabulary(
    const std::vector<DescriptorMatrix>& training_descriptors,
    int vocab_size,
    int seed);

std::vector<double> compute_image_bow_histogram(
    const DescriptorMatrix& descriptors,
    const DescriptorMatrix& vocabulary);
```

The first function produces the shared visual vocabulary. The second converts
one image's local descriptors into a fixed-length histogram.

### `src/vocabulary.cpp`

[`src/vocabulary.cpp`](src/vocabulary.cpp) implements the visual-word
representation.

#### `compute_visual_vocabulary`

The function:

1. Validates that input matrices exist and `K` is positive.
2. Finds a non-empty matrix to determine the descriptor dimension.
3. Requires all non-empty matrices to have equal dimensions.
4. Counts all sampled descriptor rows.
5. Rejects input with fewer than `K` descriptors.
6. Concatenates all training descriptors into one matrix.
7. Runs the project's k-means (k-means++ initialization, at most 50 Lloyd
   iterations) with the supplied seed. The cap of 50 gets within 0.1% of the
   fully converged clustering error on the training descriptors, which would
   need 240-300 iterations.
8. Returns the `K × 128` center matrix.

The resulting centers are the visual words shared by every image in the
evaluation set.

#### `compute_image_bow_histogram`

For one image, the function:

1. Rejects an empty descriptor matrix with `std::invalid_argument`.
2. Validates the vocabulary and descriptor dimensions.
3. Creates a zero histogram with `K` bins.
4. Assigns every descriptor to its nearest vocabulary center by Euclidean
   distance with `nearest_center`.
5. Increments the selected visual-word bin.
6. Divides every bin by the number of descriptors.
7. Returns a normalized `std::vector<double>`.

For a non-empty image, the histogram values sum to one. Empty descriptors no
longer produce an all-zero histogram: the function throws explicitly so callers
cannot accidentally insert an invalid zero vector into a database or query
collection. `main.cpp` already rejects failed extraction results before calling
this function. The vocabulary must not be empty, and descriptors must have
the same number of columns as the vocabulary.

## Pipeline module

[`include/pipeline.hpp`](include/pipeline.hpp) and
[`src/pipeline.cpp`](src/pipeline.cpp) contain reusable preparation helpers.
They keep filesystem and image-processing mechanics out of `main.cpp`.

### `extract_training_descriptors`

This helper receives training image paths, extracts SIFT descriptors, samples
up to `S_max` descriptors per successful image using the supplied random
generator, and returns sampled matrices, success/failure counts, and failed
paths. It does not print or terminate; the caller decides how to report
failures and whether zero successful images is fatal.

### `extract_bow_vectors`

This helper receives `ImageRecord` values and a trained vocabulary. For each
usable image it extracts SIFT descriptors and computes a normalized BoVW
vector. It returns vectors and identifiers in matching order, plus records
that failed extraction.

An optional successful-record limit supports the MIRFlickr rule: scanning stops
after the requested number of usable distractors has been collected. A limit
of zero processes every record. The helper does not decide whether failed
records are warnings or fatal errors.

### `extract_bow_vector_sets`

The same operation for several vocabularies at once: each image's SIFT
descriptors are extracted once and turned into one histogram per vocabulary,
so comparing vocabulary sizes does not repeat the extraction.
`MultiBowVectorResult::vectors[v][i]` is the histogram of the `i`-th usable
image under vocabulary `v`, and `record_indices[i]` is that image's position in
the input records, which lets a caller tell how many images were skipped before
any prefix of the usable ones. `extract_bow_vectors` is a wrapper around it
with a single vocabulary.

## Search Implementation and Evaluation Module

The public interface is the single header
[`include/search.hpp`](include/search.hpp). Its implementation is split by
method under [`src/search/`](src/search/):

| File | Contents |
|---|---|
| `common.hpp` / `common.cpp` | Private helpers shared by all methods, in the `image_search::detail` namespace: squared Euclidean distance (a wrapper over the k-means module's `squared_distance`, so exact distances in every method and IVFPQ's lookup tables use the same four-accumulator loop and query times are comparable), nearest-center lookup, `trainIVFClusters`, which builds the coarse clusters of IVFFlat and IVFPQ as in the lecture's IVF build step (the shared k-means runs on a seeded random sample X' of ⌈√n⌉ BoVW vectors, then all n vectors, the sampled ones included, are assigned to their nearest center, so no database image is left out of the inverted lists; `kclusters` is capped at the sample size), the `DatabaseIndex` base that validates the database, `buildResult`, which ranks candidates by exact distance, skipping duplicate candidates with a per-query flag array, and `buildRankedResult`, which takes an already-ranked list (IVFPQ's ADC ranking), stops the query timer, and computes the metrics. Also defines the public `computeEuclideanDistance`. |
| `exact.cpp` | `ExactSearch` |
| `lsh.cpp` | `LSHSearch` |
| `hypercube.cpp` | `HypercubeSearch` |
| `ivfflat.cpp` | `IVFFlatSearch` |
| `ivfpq.cpp` | `IVFPQSearch` |
| `output.cpp` | `evaluateAndWriteResults`, which writes the result file |

`common.hpp` is kept in `src/` rather than `include/` because it is an
internal header of the search module and not part of its public interface.

This module includes the implementation of **Exact Search** (`ExactSearch` class) and evaluation metrics:

- **Exact Search**: Computes the Euclidean distance between each query BoW vector and all database BoW vectors (`database_bows`), sorts them in ascending order of distance, and selects the top-10 results.
- **Evaluation & Metrics**: Automatically evaluates each query against the ground-truth relevant HPatches images ($I_2$ through $I_6$ of the sequence) to compute:
  - `Recall@5` and `Recall@10`
  - `AP@10` (Average Precision)
  - Query execution time in milliseconds. Timing starts immediately before
    the index query and stops after the ranked top-10 identifiers are
    available; ground-truth construction and metric calculations are excluded.
- **Output Generation**: Computes overall averages across all queries (`Mean Recall@5`, `Mean Recall@10`, `mAP@10`, and `Average query time`) and writes the final evaluation report to the path specified by `-o` adhering strictly to the required output format.

## `src/main.cpp`

[`src/main.cpp`](src/main.cpp) is the application entry point and coordinates
the implemented stages and failure policy. Descriptor extraction and BoVW
generation are delegated to the pipeline module, while `main.cpp` combines
returned vectors and handles application-level conditions.

### `Args`

The private `Args` structure stores:

- Dataset paths: `-hp`, `-mir`, and `-split`
- Vocabulary/training parameters: `-vocab` and `-S`
- Distractor count: `-D`
- Evaluation set: `-set`
- Output path: `-o`
- Random seed: `-seed`
- Search method and future method-specific parameters

The parser recognizes exactly one method flag:

```text
-exact
-lsh
-hypercube
-ivfflat
-ivfpq
```

`parse_int` and `parse_double` reject malformed values such as `256abc`.
Duplicate options, missing values, unknown options, multiple method flags, and
method-incompatible options terminate with an error.

The current validation also enforces:

- `-vocab` present in single-run mode (optional with `-experiment`)
- Positive `-vocab` (when given), `-S`, and `-D`
- `-D` in `{1000, 5000, 10000, 25000}`
- `-set` equal to `validation` or `test`
- Finite, positive `-w`
- Positive method parameters
- Positive explicit `-kclusters`, `-nprobe`, and `-M`
- `-vocab` divisible by `-M` for IVFPQ

IVF parameters use `-1` as an internal “not explicitly provided” value.

### `load_dataset_splits`

`load_dataset_splits` is the single split-file parser used by both the
training setup and evaluation-dataset construction. It requires exactly:

```text
70 train sequences
23 validation sequences
23 test sequences
```

The function returns a map from subset name to sequence names and throws
descriptive exceptions for missing files, malformed lines, invalid subset
names, duplicate sequences, or incorrect split counts.

`main.cpp` converts those exceptions into application-level error messages.
The evaluation API uses the same loader when selecting the requested
validation or test sequences, so split-file format and validation rules are
not duplicated across modules.

### Main execution sequence

The current `main` function performs these stages:

1. Parse and validate command-line arguments.
2. Load and validate the split counts.
3. Generate all six training-image paths for the training sequences.
4. Create a seeded Mersenne Twister generator.
5. Delegate training extraction and descriptor sampling to the pipeline module.
6. Count failures and abort clearly if every training image fails.
7. Train the visual vocabulary.
8. Construct query, HPatches database, and MIRFlickr distractor records.
9. Delegate HPatches database vectorization to the pipeline module.
10. Delegate MIRFlickr vectorization with a successful-image limit of `D`.
11. Resolve IVF defaults after the final database size is known.
12. Delegate query vectorization and reject any failed query.
13. Print the number of prepared database and query vectors.

Unreadable or descriptor-less training images generate warnings and are
excluded. Invalid HPatches database images are also skipped. Query failures are
fatal because queries cannot silently disappear from evaluation. If fewer than
`D` usable MIRFlickr images exist, the program prints a warning with the actual
number and the number of skipped images, and continues with all usable images.
After writing the result file it also prints how many queries returned fewer
than ten results, since the output format has no field for that count but the
experimental study must report it.

The final prepared data currently consists of:

```text
database_bows : vector<vector<double>>
database_ids  : vector<string>
query_bows    : vector<vector<double>>
query_ids     : vector<string>
```

The selected search method consumes these vectors after preparation. Approximate
methods use their index-specific candidate set, ranked by exact Euclidean
distance for LSH, Hypercube, and IVFFlat and by ADC distance for IVFPQ; the
exact method evaluates every database vector.

### Deferred IVF defaults

After database construction, `n_database` is the number of usable HPatches
database vectors plus the selected usable MIRFlickr vectors. For IVFFlat and
IVFPQ:

```text
default kclusters = max(5, round(n_database^(1/4)))
default nprobe    = 5
k-means sample    = ceil(sqrt(n_database)) vectors
```

These values are resolved after invalid images have been removed, so they are
based on the actual index size rather than the nominal requested size.

The assignment's `kclusters` default of ⌈√n⌉ was withdrawn by the professor in
the course Q&A: ⌈√n⌉ is the size of the random k-means training sample, not
the number of clusters, and the number of clusters should be smaller, in the
range 5 to n^(1/4), chosen on the validation set. The default
`max(5, round(n^(1/4)))` gives 6, 8, 10, and 13 clusters for the nominal
database sizes 1115, 5115, 10115, and 25115. The default `nprobe=5` is the
assignment's; with 5 or 6 clusters it probes most or all of the lists.

## Command-line usage

### Build

```bash
# Required Makefile entry point
make

# Equivalent direct CMake workflow (Release unless a build type is given)
cmake -S . -B build
cmake --build build -j

# Debug build with sanitizers
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j
```

### Example

```bash
./build/search \
  -hp path/to/hpatches \
  -mir path/to/MIRFlickr-25K \
  -split split.txt \
  -vocab 256 \
  -S 500 \
  -D 10000 \
  -set validation \
  -o output.txt \
  -ivfflat \
  -seed 1
```

### Experiment mode

Experiment mode reuses the common dataset, vocabulary, and seed arguments and
writes Markdown-ready aggregate tables:

```bash
./build/search \
  -hp path/to/hpatches \
  -mir path/to/MIRFlickr-25K \
  -split split.txt \
  -S 500 \
  -seed 1 \
  -o experimental_results.md \
  -experiment
```

Without `-vocab`, experiment mode compares the vocabulary sizes
`K` ∈ {128, 256, 512, 1024}; with `-vocab K` it studies that single size. For
every `K`, it tunes the approximate methods on validation with `D=10000`, then
evaluates Exact, LSH, Hypercube, IVFFlat, and IVFPQ on the test set for all four
required distractor sizes.

All preprocessing is done once per run:

- The training descriptors are sampled once, and every vocabulary is trained
  on the same sample with the same seed.
- The SIFT descriptors of every validation, test, and MIRFlickr image are
  extracted once and encoded with every vocabulary
  (`extract_bow_vector_sets`).
- The first 25000 usable MIRFlickr images are encoded once. Validation and
  every test `D` use the first `D` of them, which are exactly the first `D`
  usable images in ascending order, as the assignment requires.
- An IVF index is trained once per prepared database and `kclusters` value and
  reused for every `nprobe` value (`withProbes`), since `nprobe` only affects
  the search.

The vocabulary is selected with the same kind of rule as the method
parameters: the smallest `K` whose exact-search validation mAP@10 is within
0.01 of the best `K`, since a smaller `K` gives shorter vectors and cheaper
searches. Test results are reported for every `K`, the selected one is marked
in the vocabulary comparison table.

Validation grid (the parameters that mainly control the quality/time
trade-off are varied, the others are kept at fixed values):

| Method | Varied | Fixed |
|---|---|---|
| LSH | `w` ∈ {0.01, 0.025, 0.05, 0.1}, `L` ∈ {3, 5, 7, 9} | `k=4` |
| Hypercube | `M` ∈ {250, 500, 1000, 2000}, `probes` ∈ {5, 10, 20, 40} | `kproj=10`, `w=0.05` |
| IVFFlat | `kclusters` ∈ {5, 6, 8, 10}, `nprobe` ∈ {1, 2, 3, 5} | - |
| IVFPQ | `kclusters` ∈ {5, 6, 8, 10}, `nprobe` ∈ {1, 2, 3, 5} | `M=16`, `nbits=8` |

The `kclusters` values cover the professor's range of 5 to n^(1/4), about 10
for the validation database of 10115 vectors, and every `nprobe` is at most the
smallest `kclusters`. The selected configuration, `kclusters` included, is
used unchanged for every test `D`, since the test parameters are the ones
chosen on validation. The k-means sample still grows with the database
(⌈√n⌉ vectors), and at `D=1000` (n^(1/4) ≈ 5.8) a selected `kclusters` above 6
lies outside that range.

Selection rule: for each method, every configuration whose validation mAP@10
is within 0.01 of that method's best counts as equally accurate, and among
them the one with the fewest mean candidates is selected, the first in grid
order on ties. With 23 validation queries, a change of 0.23 in a single query's
AP already moves mAP@10 by 0.01, so smaller differences are not meaningful.
Candidates are the distinct database vectors whose distance to the query was
computed (exact distances, or ADC distances for IVFPQ), recorded in
`SearchResult::candidates_examined`. This count is deterministic, unlike the
measured query time, so the same seed always selects the same parameters.

Query times in experiment mode are measured with a warm-up and repetitions.
For every configuration, the whole query set first runs once untimed, so the
first queries do not pay for cold caches. Then 5 timed passes run over the
query set (`kTimedPasses`). Each query's time is the median of its 5 runs, and
the reported mean query time is the mean of these medians. Whole passes are
repeated rather than one query several times in a row, so a query does not
find its own candidates still in the cache. The median ignores single runs
slowed down by the operating system. The searches are deterministic, so the
rankings and metrics are those of the first timed pass. Single-run mode
searches each query once and reports that time.
The report contains:

- a vocabulary comparison table with the exact-search validation metrics of
  every `K`, the selected `K` marked,
- for every `K`, the selected parameters of every approximate method,
- a validation table per method with Recall@5, Recall@10, mAP@10, mean query
  time, mean candidates, and the number of queries with fewer than ten results
  for every configuration, the selected one marked, plus exact search as the
  reference,
- for every `D`, the actual number of usable MIRFlickr distractors, the
  database size, and overall, illumination, and viewpoint aggregates of every
  method on the test set, with the same columns.

### Common options

| Option | Meaning | Current default |
|---|---|---|
| `-hp <path>` | HPatches root directory | required |
| `-mir <path>` | MIRFlickr root directory | required |
| `-split <file>` | Training/validation/test split file | required |
| `-vocab <int>` | Number `K` of vocabulary centers | required, except in experiment mode, where omitting it compares 128, 256, 512, 1024 |
| `-S <int>` | Maximum sampled training descriptors per image | `500` |
| `-D <int>` | Usable MIRFlickr distractors | `1000` |
| `-set validation\|test` | Evaluation split | `validation` |
| `-o <file>` | Intended output path | `output.txt` |
| `-seed <int>` | Random seed | `1` |

Allowed values for `-D` are `1000`, `5000`, `10000`, and `25000`.

### Search-method options

| Method | Accepted parameters |
|---|---|
| `-exact` | No method-specific parameters |
| `-lsh` | `-k`, `-L`, `-w` |
| `-hypercube` | `-kproj`, `-w`, `-M`, `-probes` |
| `-ivfflat` | `-kclusters`, `-nprobe` |
| `-ivfpq` | `-kclusters`, `-nprobe`, `-M`, `-nbits` |

The parser stores these values, validates their compatibility, and passes them
to the selected index.

## Current implementation boundary

Implemented:

- Dataset split validation and path construction
- SIFT extraction
- Reproducible descriptor sampling
- Seeded k-means vocabulary training (own k-means++ and Lloyd implementation)
- Normalized BoVW histogram construction
- Robust handling of unreadable and descriptor-less images
- Dynamic usable-MIRFlickr selection
- Dynamic IVF default calculation

Implemented search methods:

- Exact exhaustive Euclidean search
- LSH with Gaussian p-stable projections and multiple hash tables. Each table
  combines its `k` integer bucket hashes using seeded random coefficients:
  `g(p) = sum(r_i h_i(p)) mod M`, with a fixed large modulus.
- Hypercube search with projection vertices, Hamming-distance probes, and a
  maximum candidate limit. A query generates only the first `probes` vertices,
  in increasing Hamming distance from its own vertex and ascending vertex
  number within a distance, instead of sorting all `2^kproj` vertices, so the
  probing cost depends on `probes` and not on `kproj`
- IVFFlat with k-means coarse centroids and `nprobe` lists. `withProbes`
  returns a copy that shares the trained index with a different `nprobe`; the
  same holds for IVFPQ.
- IVFPQ with IVF coarse lists and product-quantized residuals. A query builds
  one `M × 2^nbits` lookup table per probed list and returns the ten stored
  vectors with the smallest asymmetric distance (the sum of `M` table lookups),
  without reranking by exact distance, so IVFPQ trades accuracy for speed
  relative to IVFFlat. Codes are stored as one flat array of 16-bit indices.

In single-run mode, all methods write the assignment's result file through
`-o`. Exact search ranks the complete database. Approximate indexes rank only
the candidates returned by their index (IVFPQ by ADC distance, the others by
exact distance); they do not fall back to exhaustive
database ranking when no candidates are returned. Consequently, an
approximate result may contain fewer than ten entries, and missing positions
are treated as non-relevant when computing the metrics.

Search constructors reject empty identifiers, inconsistent dimensions, and
non-finite database values. Query dimensions are checked as an internal
assertion and also reported as an exception in release builds; non-finite
query values are rejected explicitly. An empty approximate candidate set is
not an API error because the assignment defines missing result positions as
non-relevant.

Experiment mode writes a separate Markdown-ready report through `-o`, including
the validation results of every configuration, the selected parameters, and
aggregate test results for each required distractor size, with the actual
number of usable distractors and the number of queries with fewer than ten
results.

## Development configuration

[`.vscode/c_cpp_properties.json`](.vscode/c_cpp_properties.json) configures
VS Code IntelliSense for:

- C++20
- OpenCV headers under `/usr/include/opencv4`
- The project include tree

The CMake configuration remains the authoritative build configuration. The
`.vscode/settings.json` file contains editor-specific settings and does not
participate in compilation.

## Input assumptions

The code expects:

- HPatches sequences with `1.ppm` through `6.ppm`
- MIRFlickr image files in the supplied directory
- A split file with one sequence and subset per line
- OpenCV 4.4 or newer, where SIFT is part of `features2d`
- A Linux environment with a C++20 compiler

For a fixed seed and unchanged input ordering, training descriptor sampling and
vocabulary initialization are reproducible. Filesystem access and image
decoding can still affect which images are usable if the input data changes.
