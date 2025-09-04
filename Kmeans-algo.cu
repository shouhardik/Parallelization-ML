#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define D   2      // point dim
#define K   10     // clusters
#define TPB 32     // threads per block

// --- error check macro ------------------------------------------------------
#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while (0)

// Euclidean distance on device (float)
__device__ float distance2d(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1, dy = y2 - y1;
    return sqrtf(dx*dx + dy*dy);
}

// Each thread assigns one point
__global__ void kMeansClusterAssignment(const float* __restrict__ d_datapoints,
                                        int*   __restrict__ d_clust_assn,
                                        const float* __restrict__ d_centroids,
                                        int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = d_datapoints[2*idx + 0];
    float py = d_datapoints[2*idx + 1];

    float best_dist = FLT_MAX;
    int   best_id   = -1;
    #pragma unroll
    for (int c = 0; c < K; ++c) {
        float cx = d_centroids[2*c + 0];
        float cy = d_centroids[2*c + 1];
        float d  = distance2d(px, py, cx, cy);
        if (d < best_dist) { best_dist = d; best_id = c; }
    }
    d_clust_assn[idx] = best_id;
}

// CPU centroid update (means)
void kMeansCentroidUpdate(const float* h_datapoints, const int* h_clust_assn,
                          float* h_centroids, int* h_clust_sizes, int N) {
    float sums[2*K] = {0.0f};

    for (int i = 0; i < N; ++i) {
        int cid = h_clust_assn[i];
        if (cid >= 0 && cid < K) {
            sums[2*cid + 0] += h_datapoints[2*i + 0];
            sums[2*cid + 1] += h_datapoints[2*i + 1];
            h_clust_sizes[cid] += 1;
        }
    }
    for (int c = 0; c < K; ++c) {
        if (h_clust_sizes[c] > 0) {
            h_centroids[2*c + 0] = sums[2*c + 0] / h_clust_sizes[c];
            h_centroids[2*c + 1] = sums[2*c + 1] / h_clust_sizes[c];
        }
        // if empty cluster, keep previous centroid (no change)
    }
}

// Read "x y" pairs
bool Read_from_file(float* h_datapoints, const string& input_file) {
    FILE* file = fopen(input_file.c_str(), "r");
    if (!file) return false;
    int d = 0;
    while (!feof(file)) {
        float x, y;
        if (fscanf(file, "%f %f", &x, &y) != 2) break;
        h_datapoints[2*d + 0] = x;
        h_datapoints[2*d + 1] = y;
        ++d;
    }
    fclose(file);
    return true;
}

void centroid_init(const float* h_datapoints, float* h_centroids, int N) {
    for (int i = 0; i < K; ++i) {
        int span = max(1, N / K);
        int base = i * span;
        int r    = rand() % span;
        int idx  = min(base + r, N - 1);
        h_centroids[2*i + 0] = h_datapoints[2*idx + 0];
        h_centroids[2*i + 1] = h_datapoints[2*idx + 1];
    }
}

void input_user(string* infile_name, int* num, int* epochs) {
    cout << "Number (int) of points (e.g., 100, 1000, 10000, ...):\n";
    cin  >> *num;
    switch (*num) {
        case 100:     *infile_name = "points_100.txt"; break;
        case 500:     *infile_name = "points_500.txt"; break;
        case 1000:    *infile_name = "points_1_000.txt"; break;
        case 10000:   *infile_name = "points_10_000.txt"; break;
        case 50000:   *infile_name = "points_50_000.txt"; break;
        case 100000:  *infile_name = "points_100_000.txt"; break;
        case 250000:  *infile_name = "points_250_000.txt"; break;
        case 1000000: *infile_name = "points_1_000_000.txt"; break;
        default:
            *infile_name = "points_100.txt";
            cout << "Dataset not found; using points_100.txt\n";
    }
    cout << "Epochs (iterations):\n";
    cin  >> *epochs;
}

int main() {
    srand(5);

    string input_file;
    int N, MAX_ITER;
    input_user(&input_file, &N, &MAX_ITER);

    // Host allocations
    float* h_datapoints = (float*)malloc(D * N * sizeof(float));
    float* h_centroids  = (float*)malloc(D * K * sizeof(float));
    int*   h_clust_assn = (int*)  malloc(N * sizeof(int));
    int*   h_clust_sizes= (int*)  malloc(K * sizeof(int));

    if (!Read_from_file(h_datapoints, input_file)) {
        cerr << "Error opening " << input_file << "\n";
        return 1;
    }
    centroid_init(h_datapoints, h_centroids, N);
    cout << "Initialization of " << K << " centroids:\n";
    for (int c = 0; c < K; ++c)
        cout << "(" << h_centroids[2*c] << ", " << h_centroids[2*c+1] << ")\n";

    memset(h_clust_sizes, 0, K * sizeof(int));

    // Device allocations
    float* d_datapoints = nullptr;
    float* d_centroids  = nullptr;
    int*   d_clust_assn = nullptr;

    CUDA_CHECK(cudaMalloc(&d_datapoints, D * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids,  D * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_clust_assn, N * sizeof(int)));

    // Initial H2D copies
    auto start_cp0 = high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_datapoints, h_datapoints, D * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids,  h_centroids,  D * K * sizeof(float), cudaMemcpyHostToDevice));
    auto stop_cp0  = high_resolution_clock::now();
    cout << "Initial H2D time: "
         << duration_cast<microseconds>(stop_cp0 - start_cp0).count() << " us\n";

    float time_assign = 0.0f, time_d2h = 0.0f, time_h2d = 0.0f;

    auto while_start = high_resolution_clock::now();
    for (int it = 0; it < MAX_ITER; ++it) {

        // Assignment (GPU)
        auto a0 = high_resolution_clock::now();
        dim3 block(TPB), grid((N + TPB - 1) / TPB);
        kMeansClusterAssignment<<<grid, block>>>(d_datapoints, d_clust_assn, d_centroids, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // ensure kernel finished before timing
        auto a1 = high_resolution_clock::now();
        time_assign += duration_cast<microseconds>(a1 - a0).count();

        // Bring assignments back (D2H)
        auto d0 = high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(h_clust_assn, d_clust_assn, N * sizeof(int), cudaMemcpyDeviceToHost));
        auto d1 = high_resolution_clock::now();
        time_d2h += duration_cast<microseconds>(d1 - d0).count();

        // Reset sizes for this iteration
        memset(h_clust_sizes, 0, K * sizeof(int));

        // Recompute centroids (CPU)
        kMeansCentroidUpdate(h_datapoints, h_clust_assn, h_centroids, h_clust_sizes, N);

        // Push new centroids (H2D)
        auto h0 = high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, D * K * sizeof(float), cudaMemcpyHostToDevice));
        auto h1 = high_resolution_clock::now();
        time_h2d += duration_cast<microseconds>(h1 - h0).count();
    }
    auto while_stop = high_resolution_clock::now();

    cout << "Time for " << MAX_ITER << " iterations: "
         << duration_cast<microseconds>(while_stop - while_start).count() << " us\n";
    cout << "Avg assign (GPU): " << time_assign / MAX_ITER << " us\n";
    cout << "Avg D2H (assignments): " << time_d2h / MAX_ITER << " us\n";
    cout << "Avg H2D (centroids):   " << time_h2d / MAX_ITER << " us\n";

    cout << "Final centroids:\n";
    for (int c = 0; c < K; ++c)
        cout << c << ": (" << h_centroids[2*c] << ", " << h_centroids[2*c+1] << ")\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_datapoints));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_clust_assn));
    free(h_datapoints);
    free(h_centroids);
    free(h_clust_assn);
    free(h_clust_sizes);
    return 0;
}
