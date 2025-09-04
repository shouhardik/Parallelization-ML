
// kmeans_omp.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define NUM_THREADS     8
#define DIMENSIONS      2
#define MAX_ITERATIONS  1000

typedef struct {
    double x;
    double y;
} Point;

static inline double dist2(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return dx*dx + dy*dy;  // squared distance (no sqrt needed)
}

void read_points_from_file(const char *filename, int num_points, Point *points) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Unable to open file %s.\n", filename);
        exit(1);
    }
    for (int i = 0; i < num_points; i++) {
        if (fscanf(file, "%lf %lf", &points[i].x, &points[i].y) != 2) {
            fprintf(stderr, "File %s ended early at line %d.\n", filename, i+1);
            fclose(file);
            exit(1);
        }
    }
    fclose(file);
}

void k_means_clustering(const char *label, int num_points, Point *points, int num_clusters) {
    // Centroids: initialize to first K points (consider k-means++ for quality)
    Point centroids[num_clusters];
    #pragma omp parallel for
    for (int i = 0; i < num_clusters; i++) {
        centroids[i] = points[i];
    }

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Per-iteration global accumulators
        int    cluster_counts[num_clusters]; memset(cluster_counts, 0, sizeof(cluster_counts));
        double sumx[num_clusters];           memset(sumx, 0, sizeof(sumx));
        double sumy[num_clusters];           memset(sumy, 0, sizeof(sumy));

        // Parallel region with thread-local accumulators (avoid atomics in inner loop)
        #pragma omp parallel
        {
            int    lc[num_clusters]; memset(lc, 0, sizeof(lc));
            double lx[num_clusters]; memset(lx, 0, sizeof(lx));
            double ly[num_clusters]; memset(ly, 0, sizeof(ly));

            #pragma omp for nowait
            for (int i = 0; i < num_points; i++) {
                int best = -1;
                double bestd = INFINITY;

                for (int j = 0; j < num_clusters; j++) {
                    double d = dist2(points[i], centroids[j]);
                    if (d < bestd) { bestd = d; best = j; }
                }

                // Thread-local accumulation
                lc[best] += 1;
                lx[best] += points[i].x;
                ly[best] += points[i].y;
            }

            // Merge once per thread
            #pragma omp critical
            {
                for (int j = 0; j < num_clusters; j++) {
                    cluster_counts[j] += lc[j];
                    sumx[j]          += lx[j];
                    sumy[j]          += ly[j];
                }
            }
        } // end parallel

        // Update centroids & check convergence
        int changed_any = 0;
        #pragma omp parallel for reduction(|:changed_any)
        for (int j = 0; j < num_clusters; j++) {
            if (cluster_counts[j] > 0) {
                Point newc = (Point){ sumx[j] / cluster_counts[j],
                                      sumy[j] / cluster_counts[j] };
                if (fabs(newc.x - centroids[j].x) > 1e-4 ||
                    fabs(newc.y - centroids[j].y) > 1e-4) {
                    centroids[j] = newc;
                    changed_any |= 1;
                }
            }
            // else: keep previous centroid when cluster empty
        }

        if (!changed_any) break; // converged
    }

    // Print final centroids
    printf("Final Centroids for %s:\n", label);
    for (int i = 0; i < num_clusters; i++) {
        printf("Centroid %d: %.4lf %.4lf\n", i + 1, centroids[i].x, centroids[i].y);
    }
}

int main(void) {
    omp_set_num_threads(NUM_THREADS);

    const char *file_names[] = {
        "points_250_000.txt",
        "points_50_000.txt",
        "points_500.txt",
        "points_100.txt",
        "points_1_000_000.txt",
        "points_100_000.txt",
        "points_10_000.txt",
        "points_1_000.txt"
    };
    const int num_points[] = {
        250000, 50000, 500, 100, 1000000, 100000, 10000, 1000
    };

    const int files = (int)(sizeof(file_names) / sizeof(file_names[0]));

    for (int f = 0; f < files; f++) {
        char path[256];
        snprintf(path, sizeof(path), "%s", file_names[f]);

        int n = num_points[f];
        Point *points = (Point *)malloc((size_t)n * sizeof(Point));
        if (!points) {
            fprintf(stderr, "Memory allocation failed for %d points.\n", n);
            return 1;
        }

        read_points_from_file(path, n, points);

        double t0 = omp_get_wtime();
        k_means_clustering(path, n, points, 10); // K = 10 clusters
        double t1 = omp_get_wtime();

        printf("Execution Time for %s: %.6f seconds\n\n", path, t1 - t0);
        free(points);
    }
    return 0;
}
