# Image Search with SIFT and Bag of Visual Words

This project is the current implementation of the first algorithmic-problems
assignment. It prepares an image-retrieval dataset using SIFT local descriptors,
learns a visual vocabulary with k-means, and converts images into fixed-size
Bag-of-Visual-Words (BoVW) vectors.

The code currently reaches the point where the training vocabulary, database
vectors, and query vectors have been prepared. The nearest-neighbor indexes,
ranking, evaluation metrics, and result-file generation are not yet implemented.
The command-line parser already accepts the assignment's five method names and
their parameters so that the later search implementations can be connected
without changing the data-preparation pipeline.

## Project structure

```text
.
├── CMakeLists.txt
├── README.md
├── split.txt
├── include/
│   ├── dataset.hpp
│   ├── image_features.hpp
│   └── vocabulary.hpp
├── src/
│   ├── dataset.cpp
│   ├── image_features.cpp
│   ├── main.cpp
│   └── vocabulary.cpp
└── .vscode/
    ├── c_cpp_properties.json
    └── settings.json
```

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
    ├── image_features module ◄──── image paths
    │       │
    │       └── SIFT descriptor matrices (cv::Mat)
    │
    └── vocabulary module ◄────── descriptor matrices
            │
            ├── visual vocabulary (K × 128 cv::Mat)
            └── normalized BoVW histograms (K doubles)
```

The headers define the public interfaces between modules. The `.cpp` files hide
the implementation details. `main.cpp` is the orchestration layer: it does not
implement SIFT or k-means itself; it calls the corresponding module functions
and handles application-level errors.

## `CMakeLists.txt`

[`CMakeLists.txt`](CMakeLists.txt) defines the build:

1. Requires CMake 3.20 or newer.
2. Declares a C++ project and requires C++20.
3. Enables `compile_commands.json` generation.
4. Locates OpenCV with `find_package(OpenCV REQUIRED)`.
5. Builds the `search` executable from all four source files.
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

- HPatches image: `v_bird/2`
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
6. Sorts the MIRFlickr paths deterministically.
7. Places every sorted MIRFlickr record in `distractor_pool`.

It returns a complete value instead of mutating vectors supplied by the caller.
Invalid set names, missing split files, malformed lines, invalid subset names,
and duplicate sequence names are reported with exceptions.

### `src/dataset.cpp`

[`src/dataset.cpp`](src/dataset.cpp) implements the dataset API.

The private `read_sequences` helper validates each non-empty split-file line as
two fields:

```text
<sequence-name> <train|validation|test>
```

It records all sequence names in a `std::set` to detect duplicates and returns
only the requested subset. The public evaluation function accepts only
`validation` and `test`, because training paths are obtained separately through
`get_training_image_paths`.

The private `list_mirflickr_images` helper scans the MIRFlickr directory,
retains regular `.jpg`, `.jpeg`, `.JPG`, and `.JPEG` files, and sorts the paths.
It does not inspect image contents; SIFT-based usability filtering is performed
later by `main.cpp`.

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
3. Requires all non-empty matrices to have equal dimensions and type `CV_32F`.
4. Counts all sampled descriptor rows.
5. Rejects input with fewer than `K` descriptors.
6. Concatenates all training descriptors into one OpenCV matrix.
7. Sets OpenCV's random state from the supplied seed.
8. Runs OpenCV k-means with k-means++ initialization.
9. Returns the `K × 128` center matrix.

The resulting centers are the visual words shared by every image in the
evaluation set.

#### `compute_image_bow_histogram`

For one image, the function:

1. Validates the vocabulary and descriptor dimensions.
2. Creates a zero histogram with `K` bins.
3. Uses a brute-force OpenCV L2 matcher to assign every descriptor to its
   nearest vocabulary center.
4. Increments the selected visual-word bin.
5. Divides every bin by the number of descriptors.
6. Returns a normalized `std::vector<double>`.

For a non-empty image, the histogram values sum to one. An empty descriptor
matrix produces an all-zero histogram, but `main.cpp` rejects empty extraction
results before calling this function for database or query images.

## `src/main.cpp`

[`src/main.cpp`](src/main.cpp) is the application entry point and currently
coordinates all implemented stages.

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

### `load_splits`

`load_splits` performs the application-level split validation before training.
It requires exactly:

```text
70 train sequences
23 validation sequences
23 test sequences
```

The function returns a map from subset name to sequence names.

The dataset module independently reads the split file when creating evaluation
records. This currently duplicates split-file parsing; the two paths validate
the same file but serve different module boundaries.

### Main execution sequence

The current `main` function performs these stages:

1. Parse and validate command-line arguments.
2. Load and validate the split counts.
3. Generate all six training-image paths for the training sequences.
4. Create a seeded Mersenne Twister generator.
5. Extract and sample SIFT descriptors from training images.
6. Count successful and failed training images.
7. Abort clearly if every training image fails.
8. Train the visual vocabulary.
9. Construct query, HPatches database, and MIRFlickr distractor records.
10. Extract descriptors and generate BoVW vectors for valid HPatches database images.
11. Scan the ordered MIRFlickr pool until `D` usable distractors have been added.
12. Resolve IVF defaults after the final database size is known.
13. Extract descriptors and generate BoVW vectors for all queries.
14. Print the number of prepared database and query vectors.

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

The search methods have not yet consumed these vectors.

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

The parser stores these values and validates their compatibility. The index
implementations that should use them are still future work.

## Current implementation boundary

Implemented:

- Dataset split validation and path construction
- SIFT extraction
- Reproducible descriptor sampling
- Seeded k-means vocabulary training
- Normalized BoVW histogram construction
- Robust handling of unreadable and descriptor-less images
- Dynamic usable-MIRFlickr selection
- Deferred IVF default calculation

Not yet implemented:

- Exact nearest-neighbor ranking
- LSH index and querying
- Hypercube index and querying
- IVFFlat index and querying
- IVFPQ index and querying
- Top-10 result generation
- Recall@5 and Recall@10
- AP@10 and mAP@10
- Query-time measurement
- Writing the required output format
- Validation-driven parameter selection

Therefore, `-o` is parsed and stored, but the current program does not yet write
the assignment's final result file.

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
