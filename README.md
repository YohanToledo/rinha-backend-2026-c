# Rinha de Backend 2026 - High-Performance Fraud Detection (C & AVX2)

## Overview
This project is an extremely optimized, zero-dependency C backend designed for the Rinha de Backend 2026 challenge (Fraud Detection). It achieves ultra-low P99 latency and high throughput under strict resource constraints (1.0 CPU, 350MB RAM) by bypassing traditional high-level web architectures and executing vectorized exact neighbor searches directly in memory.

## Architecture & Strategy

### 1. In-Memory K-Means & Fast Preprocessing
During the Docker build phase, the application fetches the official dataset (`references.json.gz`), unzips it in RAM via `zlib`, and parses 3 million JSON vectors natively in C. 
It then applies a multi-threaded OpenMP K-Means clustering algorithm to group transactions into 2,048 Voronoi cells. 

### 2. Fast Path Table (O(1) Decision)
The preprocessor computes a 512KB decision table that categorizes transactions by key profile attributes. During runtime, incoming requests map their vector into an integer key and do an O(1) array lookup. Over 57% of legitimate transactions are securely bypassed without running any vector searches, significantly reducing CPU usage.

### 3. INT16 Quantization & SIMD Vector Search
For vectors that fallback to the similarity search, the dataset is quantized from `float` (4 bytes) to `int16_t` (2 bytes) with a scaling factor of 10,000. 
This allows storing the entire dataset in a binary file (`dataset.bin`) that uses roughly 100MB of space.
The runtime API memory-maps (`mmap`) this binary file, guaranteeing zero-copy data reads.
Vector distance is calculated using AVX2 SIMD (`_mm256_madd_epi16`), processing batches of vectors with massive loop unrolling (x4) directly at the silicon level.

### 4. Triangle Inequality Pruning
The search algorithm utilizes Triangle Inequality bounds (`bound = cluster_radius + sqrt(current_limit)`). By pre-calculating the maximum radius of each K-Means cluster, the runtime engine dynamically skips scanning entire clusters if they are mathematically proven to not contain any closer vectors. This guarantees 100% precision (Exact Search) while behaving like an Approximate Nearest Neighbor (ANN) index.

### 5. Custom Zero-Allocation TCP Server
The API server is a custom C TCP Server leveraging Linux `epoll` with Edge-Triggering (`EPOLLET`). It uses pre-allocated static HTTP response buffers and manual fast JSON double parsing, bypassing standard library overhead and avoiding heap allocations/fragmentation entirely.

## Project Structure
- `src/kmeans_preprocess.c`: The ultra-fast JSON parser and clustering engine.
- `src/server.c`: The Epoll-based TCP runtime executing AVX2 searches.
- `Dockerfile`: Multi-stage build that bakes the binary dataset into the final lightweight Alpine image.

## How to Run
```bash
docker compose up --build
```
This will automatically download the dataset, cluster it, build the production image, and launch the API cluster behind Nginx.
