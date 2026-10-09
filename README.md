Implementation - THUS FAR

# Image Retrieval System - Vector Representations and the Nearest Neighbor Method

## Project Description
This system implements image search based on vector representations (Bag of Visual Words), using SIFT descriptors and k-means clustering. It supports exact and approximate search (Exact Search, LSH, Hypercube, IVFFlat, IVFPQ) for the HPatches and MIRFlickr-25K datasets.

## File Structure
* 'CMakeLists.txt': CMake configuration and build file
* 'include/image_features.hpp': Header file for extracting and sampling SIFT descriptors
* 'src/image_features.cpp': Implementation of SIFT feature extraction using OpenCV
* 'include/vocabulary.hpp': Header for creating the visual vocabulary and calculating BoVW histograms
* 'src/vocabulary.cpp': Implementation of k-means training and image representation
* 'src/main.cpp': Main program, command-line parameter parsing, and data loading
* 'split.txt': File for splitting sequences into training, validation and test sets

## Compilation Instructions
Compilation is performed using CMake with C++20 support:
```bash
cmake -S. -B build
cmake --build build -j
```
## Execution Instructions
Example of program execution:
```bash
./build/search -hp path/to/hpatches -mir path/to/mirflickr -split split.txt -vocab 256 -S 500 -D 10000 -set validation -o output.txt -ivfflat -kclusters 100 -nprobe 10 -seed 1
```
Where,
- -hp <path>: Directory containing the HPatches dataset
- -mir <path>: Directory containing the MIRFlickr-25K dataset
- -split <file>: Split file (training, validation, test)
- -vocab <int>: Number K of k-means clusters (visual vocabulary)
- -S <int>: Maximum number of SIFT descriptors per training image
- -D <int>: Number of MIRFlickr images used as distractors
- -set <validation|test>: Selection of query sets
- -o <output file>: Output file for results
- -seed <int>: Pseudo-random number seed

Method parameters:
- -k, -L, -w, -kproj, -probes, -kclusters, -nprobe, -M, -nbits
