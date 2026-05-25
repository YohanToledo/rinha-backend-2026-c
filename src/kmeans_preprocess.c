#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>
#include <zlib.h>

#define K_CLUSTERS 2048
#define MAX_ITERATIONS 10
#define TABLE_SIZE 524288

static const float mask_h_data[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f};

static inline uint32_t get_amount_bin(float val) {
    if (val >= 0.90f) return 7;
    if (val >= 0.60f) return 6;
    if (val >= 0.30f) return 5;
    if (val >= 0.15f) return 4;
    if (val >= 0.05f) return 3;
    if (val >= 0.01f) return 2;
    if (val >= 0.002f) return 1;
    return 0;
}

static inline uint32_t get_km_bin(float val) {
    if (val >= 0.90f) return 7;
    if (val >= 0.60f) return 6;
    if (val >= 0.30f) return 5;
    if (val >= 0.15f) return 4;
    if (val >= 0.05f) return 3;
    if (val >= 0.01f) return 2;
    if (val >= 0.002f) return 1;
    return 0;
}

static inline uint32_t get_tx_count_bin(float val) {
    float count = val * 20.0f;
    if (count >= 15.0f) return 7;
    if (count >= 10.0f) return 6;
    if (count >= 6.0f) return 5;
    if (count >= 4.0f) return 4;
    if (count >= 3.0f) return 3;
    if (count >= 2.0f) return 2;
    if (count >= 1.0f) return 1;
    return 0;
}

static inline uint32_t get_mcc_risk_bin(float val) {
    if (val >= 0.95f) return 7;
    if (val >= 0.85f) return 6;
    if (val >= 0.70f) return 5;
    if (val >= 0.55f) return 4;
    if (val >= 0.45f) return 3;
    if (val >= 0.30f) return 2;
    if (val >= 0.15f) return 1;
    return 0;
}

static inline uint32_t get_hour_bin(float val) {
    int hour = (int)roundf(val * 23.0f);
    if (hour >= 20) return 5;
    if (hour >= 16) return 4;
    if (hour >= 12) return 3;
    if (hour >= 8) return 2;
    if (hour >= 4) return 1;
    return 0;
}

uint32_t generate_profile_key_float(const float *vec) {
    uint32_t amount_bin = get_amount_bin(vec[0]);
    uint32_t km_bin = get_km_bin(vec[7]);
    uint32_t unknown_merchant = vec[11] > 0.5f ? 1 : 0;
    uint32_t tx_count_bin = get_tx_count_bin(vec[8]);
    uint32_t mcc_risk_bin = get_mcc_risk_bin(vec[12]);
    uint32_t online_bin = vec[9] > 0.5f ? 1 : 0;
    uint32_t present_bin = vec[10] > 0.5f ? 1 : 0;
    uint32_t is_null = vec[5] < -0.5f ? 1 : 0;
    uint32_t hour_bin = get_hour_bin(vec[3]);

    uint32_t key = 0;
    key |= (amount_bin & 7);
    key |= ((km_bin & 7) << 3);
    key |= ((unknown_merchant & 1) << 6);
    key |= ((tx_count_bin & 7) << 7);
    key |= ((mcc_risk_bin & 7) << 10);
    key |= ((online_bin & 1) << 13);
    key |= ((present_bin & 1) << 14);
    key |= ((is_null & 1) << 15);
    key |= ((hour_bin & 7) << 16);
    return key;
}

// AVX2 L2 float distance (ignores elements 14 and 15)
inline float avx2_l2_distance_16(const __m256 tl, const __m256 th, const __m256 mask_h, const float *v) {
    __m256 vl = _mm256_loadu_ps(v);
    __m256 vh = _mm256_loadu_ps(v + 8);
    
    __m256 diff_l = _mm256_sub_ps(tl, vl);
    __m256 diff_h = _mm256_sub_ps(th, vh);
    
    diff_h = _mm256_mul_ps(diff_h, mask_h);
    
    __m256 sq_l = _mm256_mul_ps(diff_l, diff_l);
    __m256 sq_h = _mm256_mul_ps(diff_h, diff_h);
    
    __m256 sum = _mm256_add_ps(sq_l, sq_h);
    
    __m128 low128 = _mm256_castps256_ps128(sum);
    __m128 high128 = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(low128, high128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    
    float dist;
    _mm_store_ss(&dist, sum128);
    return dist;
}

int main() {
    printf("--- INITIALIZING FAST C K-MEANS PREPROCESSOR ---\n");
    
    const char *input_path = "references.json.gz";
    printf("Decoding GZ JSON into RAM: %s\n", input_path);
    gzFile gz = gzopen(input_path, "rb");
    if (!gz) {
        perror("Error opening references.json.gz (run wget first!)");
        return 1;
    }

    size_t buf_capacity = 256 * 1024 * 1024; // Starts with 256MB
    char *buf = (char *)malloc(buf_capacity);
    size_t total_read = 0;
    
    printf("Decompressing GZ...\n");
    while (1) {
        if (total_read + 1048576 > buf_capacity) { 
            buf_capacity *= 2;
            buf = (char *)realloc(buf, buf_capacity);
            if (!buf) { perror("OOM while decompressing"); return 1; }
        }
        int bytes = gzread(gz, buf + total_read, 1048576);
        if (bytes < 0) { perror("gzread error"); return 1; }
        if (bytes == 0) break;
        total_read += bytes;
    }
    buf[total_read] = '\0';
    gzclose(gz);
    printf("Successfully decompressed %zu bytes.\n", total_read);

    long long max_vectors = 4000000;
    float *raw_data = (float *)malloc(max_vectors * 16 * sizeof(float));
    if (!raw_data) {
        perror("Memory error for raw_data");
        return 1;
    }

    printf("Parsing vectors and labels in pure C...\n");
    char *p = buf;
    long long N = 0;
    while ((p = strstr(p, "\"vector\":")) != NULL) {
        p = strchr(p, '[');
        if (!p) break;
        p++; // skip '['
        for(int j=0; j<14; j++) {
            raw_data[N * 16 + j] = (float)strtod(p, &p);
            if (*p == ',') p++;
        }
        
        char *l = strstr(p, "\"label\":");
        if (l) {
            l += 8;
            while(*l == ' ' || *l == '\t' || *l == '\n' || *l == '"') l++;
            if (strncmp(l, "fraud", 5) == 0) {
                raw_data[N * 16 + 14] = 1.0f;
            } else {
                raw_data[N * 16 + 14] = 0.0f;
            }
        }
        raw_data[N * 16 + 15] = 0.0f; // Padding
        N++;
        if (N >= max_vectors) break;
    }
    
    free(buf);
    printf("Parsing done! %lld vectors loaded.\n", N);
    
    printf("Generating Fast Path decision table with minimum support of 25...\n");
    uint32_t *legit_counts = (uint32_t *)calloc(TABLE_SIZE, sizeof(uint32_t));
    uint32_t *fraud_counts = (uint32_t *)calloc(TABLE_SIZE, sizeof(uint32_t));
    uint8_t *fast_path_table = (uint8_t *)malloc(TABLE_SIZE);
    
    for (long long i = 0; i < N; i++) {
        uint32_t key = generate_profile_key_float(&raw_data[i * 16]);
        int is_fraud = (raw_data[i * 16 + 14] > 0.5f);
        if (is_fraud) {
            fraud_counts[key]++;
        } else {
            legit_counts[key]++;
        }
    }
    
    long long fast_path_legit_count = 0;
    long long ambiguous_count = 0;
    
    for (int i = 0; i < TABLE_SIZE; i++) {
        uint32_t legit = legit_counts[i];
        uint32_t fraud = fraud_counts[i];
        
        // Statistical minimum support: at least 25 legitimate and ZERO frauds
        if (legit >= 25 && fraud == 0) {
            fast_path_table[i] = 0; 
            fast_path_legit_count += legit;
        } else {
            fast_path_table[i] = 255; 
            ambiguous_count++;
        }
    }
    
    printf("Fast Path: %lld legitimate vectors covered securely (%0.2f%% of dataset).\n", 
           fast_path_legit_count, ((double)fast_path_legit_count / N) * 100.0);
    
    free(legit_counts);
    free(fraud_counts);
    
    printf("Initializing centroids...\n");
    srand(42);
    float *centroids = (float *)calloc(K_CLUSTERS * 16, sizeof(float));
    uint8_t *picked = (uint8_t *)calloc(N, sizeof(uint8_t));
    for (int i = 0; i < K_CLUSTERS; i++) {
        int rand_idx;
        do {
            rand_idx = rand() % N;
        } while (picked[rand_idx]);
        picked[rand_idx] = 1;
        
        for (int j = 0; j < 14; j++) {
            centroids[i * 16 + j] = raw_data[rand_idx * 16 + j];
        }
    }
    free(picked);
    
    uint16_t *assignments = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint32_t *cluster_counts = (uint32_t *)calloc(K_CLUSTERS, sizeof(uint32_t));
    
    int num_threads = omp_get_max_threads();
    printf("Using %d OpenMP threads for K-Means computation.\n", num_threads);
    
    float *thread_sums = (float *)calloc(num_threads * K_CLUSTERS * 16, sizeof(float));
    uint32_t *thread_counts = (uint32_t *)calloc(num_threads * K_CLUSTERS, sizeof(uint32_t));
    
    __m256 mask_h = _mm256_loadu_ps(mask_h_data);
    
    double start_time = omp_get_wtime();
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        double iter_start = omp_get_wtime();
        
        memset(thread_sums, 0, num_threads * K_CLUSTERS * 16 * sizeof(float));
        memset(thread_counts, 0, num_threads * K_CLUSTERS * sizeof(uint32_t));
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float *my_sums = &thread_sums[tid * K_CLUSTERS * 16];
            uint32_t *my_counts = &thread_counts[tid * K_CLUSTERS];
            
            #pragma omp for schedule(static)
            for (long long i = 0; i < N; i++) {
                __m256 tl = _mm256_loadu_ps(&raw_data[i * 16]);
                __m256 th = _mm256_loadu_ps(&raw_data[i * 16 + 8]);
                
                float best_dist = INFINITY;
                int best_c = 0;
                
                for (int c = 0; c < K_CLUSTERS; c++) {
                    float dist = avx2_l2_distance_16(tl, th, mask_h, &centroids[c * 16]);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_c = c;
                    }
                }
                
                assignments[i] = best_c;
                my_counts[best_c]++;
                for (int j = 0; j < 14; j++) {
                    my_sums[best_c * 16 + j] += raw_data[i * 16 + j];
                }
            }
        }
        
        for (int c = 0; c < K_CLUSTERS; c++) {
            float sums[14] = {0};
            uint32_t count = 0;
            for (int t = 0; t < num_threads; t++) {
                count += thread_counts[t * K_CLUSTERS + c];
                for (int j = 0; j < 14; j++) {
                    sums[j] += thread_sums[(t * K_CLUSTERS + c) * 16 + j];
                }
            }
            cluster_counts[c] = count;
            if (count > 0) {
                for (int j = 0; j < 14; j++) {
                    centroids[c * 16 + j] = sums[j] / count;
                }
            }
        }
        
        double iter_end = omp_get_wtime();
        printf("Iteration %d/%d done in %.4f seconds.\n", iter + 1, MAX_ITERATIONS, iter_end - iter_start);
    }
    double end_time = omp_get_wtime();
    printf("K-Means completed! Total time: %.4f seconds.\n", end_time - start_time);
    
    free(thread_sums);
    free(thread_counts);
    
    uint32_t *cluster_offsets = (uint32_t *)malloc(K_CLUSTERS * sizeof(uint32_t));
    uint32_t current_offset = 0;
    for (int c = 0; c < K_CLUSTERS; c++) {
        cluster_offsets[c] = current_offset;
        current_offset += cluster_counts[c];
    }
    
    printf("Grouping and quantizing vectors into clusters...\n");
    int16_t *sorted_data_int16 = (int16_t *)malloc(N * 16 * sizeof(int16_t));
    uint32_t *current_insert_pos = (uint32_t *)malloc(K_CLUSTERS * sizeof(uint32_t));
    memcpy(current_insert_pos, cluster_offsets, K_CLUSTERS * sizeof(uint32_t));
    
    float scale = 10000.0f;
    
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < N; i++) {
        uint16_t c = assignments[i];
        
        uint32_t pos;
        #pragma omp atomic capture
        pos = current_insert_pos[c]++;
        
        for (int j = 0; j < 14; j++) {
            float val = raw_data[i * 16 + j];
            if (val == -1.0f) {
                sorted_data_int16[pos * 16 + j] = -10000;
            } else {
                sorted_data_int16[pos * 16 + j] = (int16_t)roundf(val * scale);
            }
        }
        sorted_data_int16[pos * 16 + 14] = (int16_t)raw_data[i * 16 + 14];
        sorted_data_int16[pos * 16 + 15] = 0;
    }
    
    int16_t *centroids_int16 = (int16_t *)calloc(K_CLUSTERS * 16, sizeof(int16_t));
    for (int c = 0; c < K_CLUSTERS; c++) {
        for (int j = 0; j < 14; j++) {
            float val = centroids[c * 16 + j];
            centroids_int16[c * 16 + j] = (int16_t)roundf(val * scale);
        }
        centroids_int16[c * 16 + 14] = 0;
        centroids_int16[c * 16 + 15] = 0;
    }
    
    printf("Calculating cluster max radii for Triangle Inequality pruning...\n");
    float *cluster_radii = (float *)calloc(K_CLUSTERS, sizeof(float));
    for (int c = 0; c < K_CLUSTERS; c++) {
        uint32_t max_r2 = 0;
        uint32_t count = cluster_counts[c];
        uint32_t offset = cluster_offsets[c];
        for (uint32_t i = 0; i < count; i++) {
            uint32_t pos = offset + i;
            int32_t dist = 0;
            for (int j = 0; j < 14; j++) {
                int32_t diff = (int32_t)sorted_data_int16[pos * 16 + j] - (int32_t)centroids_int16[c * 16 + j];
                dist += diff * diff;
            }
            if (dist > max_r2) {
                max_r2 = dist;
            }
        }
        cluster_radii[c] = sqrtf((float)max_r2);
    }
    
    printf("Writing dataset.bin with int16 quantized clusters and Fast Path table...\n");
    FILE *out = fopen("dataset.bin", "wb");
    if (!out) {
        perror("Error creating dataset.bin");
        return 1;
    }
    
    fwrite(fast_path_table, sizeof(uint8_t), TABLE_SIZE, out);
    
    uint32_t num_cl = K_CLUSTERS;
    fwrite(&num_cl, sizeof(uint32_t), 1, out);
    fwrite(centroids_int16, sizeof(int16_t), K_CLUSTERS * 16, out);
    fwrite(cluster_offsets, sizeof(uint32_t), K_CLUSTERS, out);
    fwrite(cluster_counts, sizeof(uint32_t), K_CLUSTERS, out);
    fwrite(cluster_radii, sizeof(float), K_CLUSTERS, out);
    fwrite(sorted_data_int16, sizeof(int16_t), N * 16, out);
    
    fclose(out);
    
    free(raw_data);
    free(centroids);
    free(assignments);
    free(cluster_counts);
    free(cluster_offsets);
    free(cluster_radii);
    free(sorted_data_int16);
    free(current_insert_pos);
    free(centroids_int16);
    free(fast_path_table);
    
    printf("Preprocessing fully successful!\n");
    return 0;
}
