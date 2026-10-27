# Image Search with SIFT and Bag of Visual Words

This project is the current implementation of the first algorithmic-problems
assignment. It prepares an image-retrieval dataset using SIFT local descriptors,
learns a visual vocabulary with k-means, and converts images into fixed-size
Bag-of-Visual-Words (BoVW) vectors.

The code prepares the training vocabulary, database vectors, and query vectors,
then searches them with exact search or one of four approximate indexes:
LSH, Hypercube, IVFFlat, and IVFPQ. Every method reranks its candidate set by
Euclidean distance, computes the retrieval metrics, and writes the result file.

## Project structure

```text
.
├── CMakeLists.txt
├── README.md
├── split.txt
├── include/
│   ├── dataset.hpp
│   ├── image_features.hpp
│   ├── pipeline.hpp
│   |── vocabulary.hpp
|   └── search.hpp 
├── src/
│   ├── experiment.cpp
│   ├── dataset.cpp
│   ├── image_features.cpp
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
    │       │       └── SIFT descriptor matrices (cv::Mat)
    │       │
    │       └── vocabulary module
    │               └── normalized BoVW histograms (K doubles)
    │
    └── vocabulary module ◄────── sampled training descriptors
            └── visual vocabulary (K × 128 cv::Mat)
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
4. Locates OpenCV with `find_package(OpenCV REQUIRED)`.
5. Builds the `search` executable from all project source files.
6. Adds `include/` to the header search path.
7. Links the executable with the OpenCV libraries.

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
SIFT descriptors must be skipped and replaced by the next image.

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

`SiftExtractionResult` keeps the descriptor matrix and status together so the
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
6. Otherwise returns the `CV_32F` descriptor matrix and `Success`.

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
3. Returns a clone when the image already has at most `S` descriptors.
4. Creates row indices for larger matrices.
5. Shuffles the indices using the caller's seeded `std::mt19937`.
6. Copies the first `S` selected rows into a new matrix.

The random generator is owned by `main.cpp`, so all training-image sampling is
reproducible for a fixed `-seed`.

## Vocabulary module

### `include/vocabulary.hpp`

[`include/vocabulary.hpp`](include/vocabulary.hpp) exposes:

```cpp
cv::Mat compute_visual_vocabulary(
    const std::vector<cv::Mat>& training_descriptors,
    int vocab_size,
    int seed);

std::vector<double> compute_image_bow_histogram(
    const cv::Mat& descriptors,
    const cv::Mat& vocabulary);
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
3. Requires the reference and all non-empty matrices to be non-empty,
   single-channel `CV_32F` matrices.
4. Requires all non-empty matrices to have equal dimensions.
5. Counts all sampled descriptor rows and rejects integer overflow.
6. Rejects input with fewer than `K` descriptors.
7. Concatenates all training descriptors into one OpenCV matrix.
8. Sets OpenCV's random state from the supplied seed.
9. Runs OpenCV k-means with k-means++ initialization.
10. Returns the `K × 128` center matrix.

The resulting centers are the visual words shared by every image in the
evaluation set.

#### `compute_image_bow_histogram`

For one image, the function:

1. Rejects an empty descriptor matrix with `std::invalid_argument`.
2. Validates the vocabulary and descriptor dimensions.
3. Creates a zero histogram with `K` bins.
4. Uses a brute-force OpenCV L2 matcher to assign every descriptor to its
   nearest vocabulary center.
5. Validates each returned nearest-word index.
6. Increments the selected visual-word bin.
7. Divides every bin by the number of descriptors.
8. Returns a normalized `std::vector<double>`.

For a non-empty image, the histogram values sum to one. Empty descriptors no
longer produce an all-zero histogram: the function throws explicitly so callers
cannot accidentally insert an invalid zero vector into a database or query
collection. `main.cpp` already rejects failed extraction results before calling
this function. The vocabulary must have positive dimensions and type `CV_32F`;
descriptor matrices must also be `CV_32F` and have the same number of columns.

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

## Search Implementation and Evaluation Module

The public interface is the single header
[`include/search.hpp`](include/search.hpp). Its implementation is split by
method under [`src/search/`](src/search/):

| File | Contents |
|---|---|
| `common.hpp` / `common.cpp` | Private helpers shared by all methods, in the `image_search::detail` namespace: squared Euclidean distance, nearest-center lookup, the k-means used by IVFFlat and IVFPQ, the `DatabaseIndex` base that validates the database, and `buildResult`, which reranks candidates and computes the metrics. Also defines the public `computeEuclideanDistance`. |
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

- Positive `-vocab`, `-S`, and `-D`
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
`D` usable MIRFlickr images exist, the program exits with an explicit error.

The final prepared data currently consists of:

```text
database_bows : vector<vector<double>>
database_ids  : vector<string>
query_bows    : vector<vector<double>>
query_ids     : vector<string>
```

The selected search method consumes these vectors after preparation. Approximate
methods use their index-specific candidate set and exact Euclidean reranking;
the exact method evaluates every database vector.

### Deferred IVF defaults

After database construction, `n_database` is the number of usable HPatches
database vectors plus the selected usable MIRFlickr vectors. For IVFFlat and
IVFPQ:

```text
default kclusters = ceil(sqrt(n_database))
default nprobe    = 5
```

These values are resolved after invalid images have been removed, so they are
based on the actual index size rather than the nominal requested size.

## Command-line usage

### Build

```bash
# Required Makefile entry point
make

# Equivalent direct CMake workflow
cmake -S . -B build
cmake --build build -j
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
  -vocab 256 \
  -S 500 \
  -seed 1 \
  -o experimental_results.md \
  -experiment
```

It tunes the approximate methods on validation with `D=10000`, selecting the
configuration with the highest mAP@10, then evaluates Exact, LSH, Hypercube,
IVFFlat, and IVFPQ on the test set for all four required distractor sizes.
The report contains overall, illumination, and viewpoint aggregates. The
validation grids focus on the principal quality/time parameters rather than
performing an exhaustive search of every parameter.

### Common options

| Option | Meaning | Current default |
|---|---|---|
| `-hp <path>` | HPatches root directory | required |
| `-mir <path>` | MIRFlickr root directory | required |
| `-split <file>` | Training/validation/test split file | required |
| `-vocab <int>` | Number `K` of vocabulary centers | required |
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
- Seeded k-means vocabulary training
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
  maximum candidate limit
- IVFFlat with k-means coarse centroids and `nprobe` lists
- IVFPQ with IVF coarse lists and product-quantized sub-vectors

In single-run mode, all methods write the assignment's result file through
`-o`. Exact search ranks the complete database. Approximate indexes rank only
the candidates returned by their index; they do not fall back to exhaustive
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
the validation-selected parameters and aggregate test results for each
required distractor size.

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
- OpenCV with SIFT support
- A Linux environment with a C++20 compiler

For a fixed seed and unchanged input ordering, training descriptor sampling and
vocabulary initialization are reproducible. Filesystem access and image
decoding can still affect which images are usable if the input data changes.
