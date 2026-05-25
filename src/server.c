#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ctype.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <immintrin.h>

#define MAX_EVENTS 1024
#define BUF_SIZE 8192
#define MAX_CLIENTS 10240

// --- CLIENT STATES ---
typedef struct {
    char buf[BUF_SIZE];
    int bytes_read;
} ClientState;

ClientState client_states[MAX_CLIENTS];

// --- MMAP DATASET VARIABLES ---
uint8_t *fast_path_table = NULL;
uint32_t num_clusters = 0;
int16_t *centroids = NULL;
uint32_t *cluster_offsets = NULL;
uint32_t *cluster_counts = NULL;
float *cluster_radii = NULL;
int16_t *sorted_data = NULL;

// --- CONFIGURATIONS ---
float mcc_risk[10000];

float norm_max_amount = 10000.0f;
float norm_max_installments = 12.0f;
float norm_amount_vs_avg_ratio = 10.0f;
float norm_max_minutes = 1440.0f;
float norm_max_km = 1000.0f;
float norm_max_tx_count_24h = 20.0f;
float norm_max_merchant_avg_amount = 10000.0f;

// --- UTILS E PARSERS ---
void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

float clamp(float val) {
    return val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val);
}

double parse_double_value(const char *ptr) {
    if (!ptr) return 0.0;
    while (*ptr && *ptr != ':') ptr++;
    if (*ptr == ':') ptr++;
    while (*ptr && (*ptr == ' ' || *ptr == '\t' || *ptr == '\r' || *ptr == '\n')) ptr++;
    
    double val = 0.0;
    int neg = 0;
    if (*ptr == '-') { neg = 1; ptr++; }
    else if (*ptr == '+') { ptr++; }
    
    while (*ptr >= '0' && *ptr <= '9') {
        val = val * 10.0 + (*ptr - '0');
        ptr++;
    }
    
    if (*ptr == '.') {
        ptr++;
        double fraction = 0.0;
        double divisor = 1.0;
        while (*ptr >= '0' && *ptr <= '9') {
            fraction = fraction * 10.0 + (*ptr - '0');
            divisor *= 10.0;
            ptr++;
        }
        val += fraction / divisor;
    }
    
    return neg ? -val : val;
}

int parse_bool_value(const char *ptr) {
    if (!ptr) return 0;
    while (*ptr && *ptr != ':') ptr++;
    if (*ptr == ':') ptr++;
    while (*ptr && (*ptr == ' ' || *ptr == '\t' || *ptr == '\r' || *ptr == '\n')) ptr++;
    if (strncmp(ptr, "true", 4) == 0) return 1;
    return 0;
}

void parse_string_value(const char *ptr, char *dest, int max_len) {
    if (!ptr) {
        dest[0] = '\0';
        return;
    }
    while (*ptr && *ptr != ':') ptr++;
    if (*ptr == ':') ptr++;
    while (*ptr && *ptr != '"') ptr++;
    if (*ptr == '"') ptr++;
    int i = 0;
    while (*ptr && *ptr != '"' && i < max_len - 1) {
        dest[i++] = *ptr++;
    }
    dest[i] = '\0';
}

long long get_epoch_seconds(int year, int month, int day, int hour, int minute, int second) {
    if (month <= 2) {
        year -= 1;
        month += 12;
    }
    long long era = (year >= 0 ? year : year - 399) / 400;
    unsigned yoe = (unsigned)(year - era * 400);
    unsigned doy = (153 * (month - 3) + 2) / 5 + day - 1;
    unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    long long days = era * 146097 + (long long)doe - 719468;
    return days * 86400LL + hour * 3600LL + minute * 60LL + second;
}

static inline int itoa_fast(int value, char *str) {
    if (value == 0) {
        str[0] = '0';
        return 1;
    }
    int i = 0;
    if (value < 0) {
        str[i++] = '-';
        value = -value;
    }
    char temp[16];
    int len = 0;
    while (value > 0) {
        temp[len++] = '0' + (value % 10);
        value /= 10;
    }
    for (int j = len - 1; j >= 0; j--) {
        str[i++] = temp[j];
    }
    return i;
}

// --- OPTIMIZED PARSERS ---

// --- CONFIGURATION LOADING ---
void load_normalization() {
    norm_max_amount = 10000.0f;
    norm_max_installments = 12.0f;
    norm_amount_vs_avg_ratio = 10.0f;
    norm_max_minutes = 1440.0f;
    norm_max_km = 1000.0f;
    norm_max_tx_count_24h = 20.0f;
    norm_max_merchant_avg_amount = 10000.0f;
}

void load_mcc_risk() {
    for (int i = 0; i < 10000; i++) {
        mcc_risk[i] = 0.5f;
    }
    
    mcc_risk[5411] = 0.15f;
    mcc_risk[5812] = 0.30f;
    mcc_risk[5912] = 0.20f;
    mcc_risk[5944] = 0.45f;
    mcc_risk[7801] = 0.80f;
    mcc_risk[7802] = 0.75f;
    mcc_risk[7995] = 0.85f;
    mcc_risk[4511] = 0.35f;
    mcc_risk[5311] = 0.25f;
    mcc_risk[5999] = 0.50f;
}

// --- FAST PATH KEY GENERATION ---
static inline uint32_t get_amount_bin(int16_t val) {
    float f_val = (float)val / 10000.0f;
    if (f_val >= 0.90f) return 7;
    if (f_val >= 0.60f) return 6;
    if (f_val >= 0.30f) return 5;
    if (f_val >= 0.15f) return 4;
    if (f_val >= 0.05f) return 3;
    if (f_val >= 0.01f) return 2;
    if (f_val >= 0.002f) return 1;
    return 0;
}

static inline uint32_t get_km_bin(int16_t val) {
    float f_val = (float)val / 10000.0f;
    if (f_val >= 0.90f) return 7;
    if (f_val >= 0.60f) return 6;
    if (f_val >= 0.30f) return 5;
    if (f_val >= 0.15f) return 4;
    if (f_val >= 0.05f) return 3;
    if (f_val >= 0.01f) return 2;
    if (f_val >= 0.002f) return 1;
    return 0;
}

static inline uint32_t get_tx_count_bin(int16_t val) {
    float f_val = (float)val / 10000.0f;
    float count = f_val * 20.0f;
    if (count >= 15.0f) return 7;
    if (count >= 10.0f) return 6;
    if (count >= 6.0f) return 5;
    if (count >= 4.0f) return 4;
    if (count >= 3.0f) return 3;
    if (count >= 2.0f) return 2;
    if (count >= 1.0f) return 1;
    return 0;
}

static inline uint32_t get_mcc_risk_bin(int16_t val) {
    float f_val = (float)val / 10000.0f;
    if (f_val >= 0.95f) return 7;
    if (f_val >= 0.85f) return 6;
    if (f_val >= 0.70f) return 5;
    if (f_val >= 0.55f) return 4;
    if (f_val >= 0.45f) return 3;
    if (f_val >= 0.30f) return 2;
    if (f_val >= 0.15f) return 1;
    return 0;
}

static inline uint32_t get_hour_bin(int16_t val) {
    float f_val = (float)val / 10000.0f;
    int hour = (int)roundf(f_val * 23.0f);
    if (hour >= 20) return 5;
    if (hour >= 16) return 4;
    if (hour >= 12) return 3;
    if (hour >= 8) return 2;
    if (hour >= 4) return 1;
    return 0;
}

uint32_t generate_profile_key(const int16_t *vec) {
    uint32_t amount_bin = get_amount_bin(vec[0]);
    uint32_t km_bin = get_km_bin(vec[7]);
    uint32_t unknown_merchant = vec[11] > 5000 ? 1 : 0;
    uint32_t tx_count_bin = get_tx_count_bin(vec[8]);
    uint32_t mcc_risk_bin = get_mcc_risk_bin(vec[12]);
    uint32_t online_bin = vec[9] > 5000 ? 1 : 0;
    uint32_t present_bin = vec[10] > 5000 ? 1 : 0;
    uint32_t is_null = vec[5] < -5000 ? 1 : 0;
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

typedef struct {
    uint32_t id;
    int32_t dist;
} ClusterDist;

static int compare_cluster_dist(const void *a, const void *b) {
    int32_t d1 = ((ClusterDist *)a)->dist;
    int32_t d2 = ((ClusterDist *)b)->dist;
    return (d1 > d2) - (d1 < d2);
}

// --- AVX2 SIMD VECTOR SEARCH WITH INT16 QUANTIZATION ---
static const int16_t mask_data[16] = {
    -1, -1, -1, -1, -1, -1, -1, -1, 
    -1, -1, -1, -1, -1, -1,  0,  0  
};

inline int32_t __attribute__((always_inline)) avx2_l2_distance_int16(const __m256i t, const __m256i mask, const int16_t *v) {
    __m256i db = _mm256_loadu_si256((const __m256i*)v);
    __m256i diff = _mm256_sub_epi16(t, db);
    diff = _mm256_and_si256(diff, mask);
    __m256i sq = _mm256_madd_epi16(diff, diff);
    
    __m128i low128 = _mm256_castsi256_si128(sq);
    __m128i high128 = _mm256_extracti128_si256(sq, 1);
    __m128i sum128 = _mm_add_epi32(low128, high128);
    
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    
    return _mm_cvtsi128_si32(sum128);
}

inline void __attribute__((always_inline)) avx2_l2_distance_int16_x4(
    const __m256i t, const __m256i mask, 
    const int16_t *v1, const int16_t *v2, const int16_t *v3, const int16_t *v4, 
    int32_t *d1, int32_t *d2, int32_t *d3, int32_t *d4) {
    
    __m256i db1 = _mm256_loadu_si256((const __m256i*)v1);
    __m256i db2 = _mm256_loadu_si256((const __m256i*)v2);
    __m256i db3 = _mm256_loadu_si256((const __m256i*)v3);
    __m256i db4 = _mm256_loadu_si256((const __m256i*)v4);
    
    __m256i diff1 = _mm256_sub_epi16(t, db1);
    __m256i diff2 = _mm256_sub_epi16(t, db2);
    __m256i diff3 = _mm256_sub_epi16(t, db3);
    __m256i diff4 = _mm256_sub_epi16(t, db4);
    
    diff1 = _mm256_and_si256(diff1, mask);
    diff2 = _mm256_and_si256(diff2, mask);
    diff3 = _mm256_and_si256(diff3, mask);
    diff4 = _mm256_and_si256(diff4, mask);
    
    __m256i sq1 = _mm256_madd_epi16(diff1, diff1);
    __m256i sq2 = _mm256_madd_epi16(diff2, diff2);
    __m256i sq3 = _mm256_madd_epi16(diff3, diff3);
    __m256i sq4 = _mm256_madd_epi16(diff4, diff4);
    
    __m128i sum1 = _mm_add_epi32(_mm256_castsi256_si128(sq1), _mm256_extracti128_si256(sq1, 1));
    __m128i sum2 = _mm_add_epi32(_mm256_castsi256_si128(sq2), _mm256_extracti128_si256(sq2, 1));
    __m128i sum3 = _mm_add_epi32(_mm256_castsi256_si128(sq3), _mm256_extracti128_si256(sq3, 1));
    __m128i sum4 = _mm_add_epi32(_mm256_castsi256_si128(sq4), _mm256_extracti128_si256(sq4, 1));
    
    __m128i sum12 = _mm_hadd_epi32(sum1, sum2);
    __m128i sum34 = _mm_hadd_epi32(sum3, sum4);
    __m128i final = _mm_hadd_epi32(sum12, sum34);
    
    *d1 = _mm_cvtsi128_si32(final);
    *d2 = _mm_extract_epi32(final, 1);
    *d3 = _mm_extract_epi32(final, 2);
    *d4 = _mm_extract_epi32(final, 3);
}

void find_k_nearest(int16_t *target_vec, int16_t *best_labels_out) {
    int32_t best_dists[5] = {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX};
    int16_t best_labels[5] = {0, 0, 0, 0, 0};

    __m256i t = _mm256_loadu_si256((const __m256i*)target_vec);
    __m256i mask = _mm256_loadu_si256((const __m256i*)mask_data);

    ClusterDist sorted_clusters[2048];

    for (uint32_t c = 0; c < num_clusters; c++) {
        int32_t dist = avx2_l2_distance_int16(t, mask, &centroids[c * 16]);
        sorted_clusters[c].id = c;
        sorted_clusters[c].dist = dist;
    }

    qsort(sorted_clusters, num_clusters, sizeof(ClusterDist), compare_cluster_dist);

    int32_t limit = best_dists[4];
    double sqrt_limit = 0.0;

    for (uint32_t idx = 0; idx < num_clusters; idx++) {
        uint32_t c = sorted_clusters[idx].id;
        int32_t c_dist = sorted_clusters[idx].dist;

        // Exact Triangle Inequality Pruning
        if (limit != INT32_MAX) {
            double r = (double)cluster_radii[c];
            double bound = r + sqrt_limit;
            if ((double)c_dist >= bound * bound) {
                continue; 
            }
        }

        uint32_t c_count = cluster_counts[c];
        uint32_t c_offset = cluster_offsets[c];
        uint32_t i = 0;

        // Massive batch processing (x4 loop unrolling)
        for (; i + 3 < c_count; i += 4) {
            uint32_t vOff1 = (c_offset + i) * 16;
            uint32_t vOff2 = (c_offset + i + 1) * 16;
            uint32_t vOff3 = (c_offset + i + 2) * 16;
            uint32_t vOff4 = (c_offset + i + 3) * 16;

            int32_t d1, d2, d3, d4;
            avx2_l2_distance_int16_x4(t, mask, 
                &sorted_data[vOff1], &sorted_data[vOff2], 
                &sorted_data[vOff3], &sorted_data[vOff4], 
                &d1, &d2, &d3, &d4);

            int32_t dists[4] = {d1, d2, d3, d4};
            uint32_t offs[4] = {vOff1, vOff2, vOff3, vOff4};

            for(int k=0; k<4; k++) {
                int32_t dist = dists[k];
                if (dist < limit) {
                    int16_t label = sorted_data[offs[k] + 14];
                    if (dist < best_dists[0]) {
                        best_dists[4] = best_dists[3]; best_labels[4] = best_labels[3];
                        best_dists[3] = best_dists[2]; best_labels[3] = best_labels[2];
                        best_dists[2] = best_dists[1]; best_labels[2] = best_labels[1];
                        best_dists[1] = best_dists[0]; best_labels[1] = best_labels[0];
                        best_dists[0] = dist; best_labels[0] = label;
                    } else if (dist < best_dists[1]) {
                        best_dists[4] = best_dists[3]; best_labels[4] = best_labels[3];
                        best_dists[3] = best_dists[2]; best_labels[3] = best_labels[2];
                        best_dists[2] = best_dists[1]; best_labels[2] = best_labels[1];
                        best_dists[1] = dist; best_labels[1] = label;
                    } else if (dist < best_dists[2]) {
                        best_dists[4] = best_dists[3]; best_labels[4] = best_labels[3];
                        best_dists[3] = best_dists[2]; best_labels[3] = best_labels[2];
                        best_dists[2] = dist; best_labels[2] = label;
                    } else if (dist < best_dists[3]) {
                        best_dists[4] = best_dists[3]; best_labels[4] = best_labels[3];
                        best_dists[3] = dist; best_labels[3] = label;
                    } else {
                        best_dists[4] = dist; best_labels[4] = label;
                    }
                    limit = best_dists[4];
                    if (limit != INT32_MAX) {
                        sqrt_limit = sqrt((double)limit);
                    }
                }
            }
        }

        // Tail processing for leftovers
        for (; i < c_count; i++) {
            uint32_t vOff = (c_offset + i) * 16;
            int32_t dist = avx2_l2_distance_int16(t, mask, &sorted_data[vOff]);
            if (dist < limit) {
                int16_t label = sorted_data[vOff + 14];
                if (dist < best_dists[0]) {
                    best_dists[4] = best_dists[3]; best_labels[4] = best_labels[3];
                    best_dists[3] = best_dists[2]; best_labels[3] = best_labels[2];
                    best_dists[2] = best_dists[1]; best_labels[2] = best_labels[1];
                    best_dists[1] = best_dists[0]; best_labels[1] = best_labels[0];
                    best_dists[0] = dist; best_labels[0] = label;
                } else if (dist < best_dists[1]) {
                    best_dists[4] = best_dists[3]; best_labels[4] = best_labels[3];
                    best_dists[3] = best_dists[2]; best_labels[3] = best_labels[2];
                    best_dists[2] = best_dists[1]; best_labels[2] = best_labels[1];
                    best_dists[1] = dist; best_labels[1] = label;
                } else if (dist < best_dists[2]) {
                    best_dists[4] = best_dists[3]; best_labels[4] = best_labels[3];
                    best_dists[3] = best_dists[2]; best_labels[3] = best_labels[2];
                    best_dists[2] = dist; best_labels[2] = label;
                } else if (dist < best_dists[3]) {
                    best_dists[4] = best_dists[3]; best_labels[4] = best_labels[3];
                    best_dists[3] = dist; best_labels[3] = label;
                } else {
                    best_dists[4] = dist; best_labels[4] = label;
                }
                limit = best_dists[4];
                if (limit != INT32_MAX) {
                    sqrt_limit = sqrt((double)limit);
                }
            }
        }
    }

    for (int i = 0; i < 5; i++) {
        best_labels_out[i] = best_labels[i];
    }
}

// --- REQUEST HANDLING ---
int process_fraud_score(const char *json) {
    float transaction_amount = (float)parse_double_value(strstr(json, "\"amount\""));
    float transaction_installments = (float)parse_double_value(strstr(json, "\"installments\""));
    
    char requested_at[32] = {0};
    parse_string_value(strstr(json, "\"requested_at\""), requested_at, sizeof(requested_at));
    
    // avg_amount sob customer
    const char *cust_ptr = strstr(json, "\"customer\"");
    float customer_avg_amount = 0.0f;
    float customer_tx_count_24h = 0.0f;
    if (cust_ptr) {
        const char *avg_ptr = strstr(cust_ptr, "\"avg_amount\"");
        if (avg_ptr) customer_avg_amount = (float)parse_double_value(avg_ptr);
        
        const char *tx_ptr = strstr(cust_ptr, "\"tx_count_24h\"");
        if (tx_ptr) customer_tx_count_24h = (float)parse_double_value(tx_ptr);
    }
    
    // avg_amount sob merchant, mcc, id
    const char *merch_ptr = strstr(json, "\"merchant\"");
    float merchant_avg_amount = 0.0f;
    char merchant_id[64] = {0};
    int merchant_mcc = 0;
    if (merch_ptr) {
        const char *avg_ptr = strstr(merch_ptr, "\"avg_amount\"");
        if (avg_ptr) merchant_avg_amount = (float)parse_double_value(avg_ptr);
        
        const char *id_ptr = strstr(merch_ptr, "\"id\"");
        if (id_ptr) parse_string_value(id_ptr, merchant_id, sizeof(merchant_id));
        
        const char *mcc_ptr = strstr(merch_ptr, "\"mcc\"");
        if (mcc_ptr) {
            char mcc_str[16] = {0};
            parse_string_value(mcc_ptr, mcc_str, sizeof(mcc_str));
            merchant_mcc = atoi(mcc_str);
        }
    }
    
    // terminal
    const char *term_ptr = strstr(json, "\"terminal\"");
    int terminal_is_online = 0;
    int terminal_card_present = 0;
    float terminal_km_from_home = 0.0f;
    if (term_ptr) {
        const char *online_ptr = strstr(term_ptr, "\"is_online\"");
        if (online_ptr) terminal_is_online = parse_bool_value(online_ptr);
        
        const char *present_ptr = strstr(term_ptr, "\"card_present\"");
        if (present_ptr) terminal_card_present = parse_bool_value(present_ptr);
        
        const char *km_ptr = strstr(term_ptr, "\"km_from_home\"");
        if (km_ptr) terminal_km_from_home = (float)parse_double_value(km_ptr);
    }
    
    // last_transaction
    const char *lt_ptr = strstr(json, "\"last_transaction\"");
    int last_transaction_is_null = 1;
    char last_transaction_timestamp[32] = {0};
    float last_transaction_km_from_current = 0.0f;
    if (lt_ptr) {
        const char *c_ptr = strchr(lt_ptr, ':');
        if (c_ptr) {
            c_ptr++;
            while (*c_ptr && (*c_ptr == ' ' || *c_ptr == '\t' || *c_ptr == '\r' || *c_ptr == '\n')) c_ptr++;
            if (strncmp(c_ptr, "null", 4) != 0) {
                last_transaction_is_null = 0;
                const char *ts_ptr = strstr(c_ptr, "\"timestamp\"");
                if (ts_ptr) parse_string_value(ts_ptr, last_transaction_timestamp, sizeof(last_transaction_timestamp));
                
                const char *km_ptr = strstr(c_ptr, "\"km_from_current\"");
                if (km_ptr) last_transaction_km_from_current = (float)parse_double_value(km_ptr);
            }
        }
    }

    // 2. VETORIZAÇÃO 100% ARITMÉTICA (16 int16_t com zeros no final)
    int16_t target_vec[16];
    float scale = 10000.0f;
    
    target_vec[0] = (int16_t)roundf(clamp(transaction_amount / norm_max_amount) * scale);
    target_vec[1] = (int16_t)roundf(clamp(transaction_installments / norm_max_installments) * scale);
    target_vec[2] = (int16_t)roundf(clamp((transaction_amount / customer_avg_amount) / norm_amount_vs_avg_ratio) * scale);
    
    // Parse requested_at
    int year = (requested_at[0]-'0')*1000 + (requested_at[1]-'0')*100 + (requested_at[2]-'0')*10 + (requested_at[3]-'0');
    int month = (requested_at[5]-'0')*10 + (requested_at[6]-'0');
    int day = (requested_at[8]-'0')*10 + (requested_at[9]-'0');
    int hour = (requested_at[11]-'0')*10 + (requested_at[12]-'0');
    int minute = (requested_at[14]-'0')*10 + (requested_at[15]-'0');
    int second = (requested_at[17]-'0')*10 + (requested_at[18]-'0');
    
    long long t1 = get_epoch_seconds(year, month, day, hour, minute, second) * 1000LL;
    
    target_vec[3] = (int16_t)roundf((hour / 23.0f) * scale);
    
    long long days_since_epoch = t1 / 86400000LL;
    int js_day = (days_since_epoch + 4) % 7;
    int rinha_day = js_day == 0 ? 6 : js_day - 1;
    target_vec[4] = (int16_t)roundf((rinha_day / 6.0f) * scale);
    
    if (last_transaction_is_null) {
        target_vec[5] = -10000;
        target_vec[6] = -10000;
    } else {
        int ly = (last_transaction_timestamp[0]-'0')*1000 + (last_transaction_timestamp[1]-'0')*100 + (last_transaction_timestamp[2]-'0')*10 + (last_transaction_timestamp[3]-'0');
        int lm = (last_transaction_timestamp[5]-'0')*10 + (last_transaction_timestamp[6]-'0');
        int ld = (last_transaction_timestamp[8]-'0')*10 + (last_transaction_timestamp[9]-'0');
        int lh = (last_transaction_timestamp[11]-'0')*10 + (last_transaction_timestamp[12]-'0');
        int lmin = (last_transaction_timestamp[14]-'0')*10 + (last_transaction_timestamp[15]-'0');
        int lsec = (last_transaction_timestamp[17]-'0')*10 + (last_transaction_timestamp[18]-'0');
        
        long long t2 = get_epoch_seconds(ly, lm, ld, lh, lmin, lsec) * 1000LL;
        float minutes = (float)((t1 - t2) / 60000.0);
        target_vec[5] = (int16_t)roundf(clamp(minutes / norm_max_minutes) * scale);
        target_vec[6] = (int16_t)roundf(clamp(last_transaction_km_from_current / norm_max_km) * scale);
    }
    
    target_vec[7] = (int16_t)roundf(clamp(terminal_km_from_home / norm_max_km) * scale);
    target_vec[8] = (int16_t)roundf(clamp(customer_tx_count_24h / norm_max_tx_count_24h) * scale);
    target_vec[9] = terminal_is_online ? 10000 : 0;
    target_vec[10] = terminal_card_present ? 10000 : 0;
    // Check known merchants
    int is_known = 0;
    char quoted_merchant[128];
    quoted_merchant[0] = '"';
    int q_idx = 1;
    while(merchant_id[q_idx-1] != '\0' && q_idx < 125) {
        quoted_merchant[q_idx] = merchant_id[q_idx-1];
        q_idx++;
    }
    quoted_merchant[q_idx++] = '"';
    quoted_merchant[q_idx] = '\0';
    const char *km_start = strstr(json, "\"known_merchants\"");
    if (km_start) {
        const char *bracket_start = strchr(km_start, '[');
        const char *bracket_end = strchr(km_start, ']');
        if (bracket_start && bracket_end && bracket_end > bracket_start) {
            char *mutable_end = (char *)bracket_end;
            char old_char = *mutable_end;
            *mutable_end = '\0';
            if (strstr(bracket_start, quoted_merchant) != NULL) {
                is_known = 1;
            }
            *mutable_end = old_char;
        }
    }
    target_vec[11] = is_known ? 0 : 10000;
    
    float risk = (merchant_mcc >= 0 && merchant_mcc < 10000) ? mcc_risk[merchant_mcc] : 0.5f;
    target_vec[12] = (int16_t)roundf(risk * scale);
    target_vec[13] = (int16_t)roundf(clamp(merchant_avg_amount / norm_max_merchant_avg_amount) * scale);
    
    target_vec[14] = 0;
    target_vec[15] = 0;

    int fraud_count = 0;

    // Fast Path O(1) bypassing
    uint32_t profile_key = generate_profile_key(target_vec);
    uint8_t decision = fast_path_table[profile_key];
    if (decision == 0) {
        fraud_count = 0;
    } else {
        int16_t top5_labels[5] = {0};
        find_k_nearest(target_vec, top5_labels);

        for (int i = 0; i < 5; i++) {
            fraud_count += top5_labels[i];
        }
    }

    return fraud_count;
}

static const char *RESP[6] = {
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}"
};
static const int RESP_LEN[6] = {130, 130, 130, 131, 131, 131};

// --- HTTP ROUTER & NETWORKING LOGIC ---

static inline void handle_http_request(int client_fd, ClientState *state, char *body, int content_length) {
    if (strncmp(state->buf, "GET /ready", 10) == 0) {
        const char *resp_ready = 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 15\r\n"
            "Connection: keep-alive\r\n"
            "\r\n"
            "{\"status\":\"ok\"}";
        write(client_fd, resp_ready, strlen(resp_ready));
    } 
    else if (strncmp(state->buf, "POST /fraud-score", 17) == 0) {
        body[content_length] = '\0';
        int fraud_count = process_fraud_score(body);
        write(client_fd, RESP[fraud_count], RESP_LEN[fraud_count]);
    } 
    else {
        const char *resp_404 = 
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Length: 0\r\n"
            "Connection: keep-alive\r\n"
            "\r\n";
        write(client_fd, resp_404, strlen(resp_404));
    }
}

static inline void accept_new_connections(int epoll_fd, int server_fd) {
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break; 
            }
            perror("accept error");
            break;
        }

        if (client_fd >= MAX_CLIENTS) {
            close(client_fd);
            continue;
        }

        set_nonblocking(client_fd);
        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        setsockopt(client_fd, IPPROTO_TCP, TCP_QUICKACK, &flag, sizeof(flag));
        client_states[client_fd].bytes_read = 0;

        struct epoll_event client_ev;
        client_ev.events = EPOLLIN | EPOLLET;
        client_ev.data.fd = client_fd;
        epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &client_ev);
    }
}

static inline void handle_client_data(int epoll_fd, int client_fd) {
    ClientState *state = &client_states[client_fd];
    int closed = 0;

    while (1) {
        int remaining = BUF_SIZE - state->bytes_read - 1;
        if (remaining <= 0) {
            state->bytes_read = 0;
            remaining = BUF_SIZE - 1;
        }

        int n = read(client_fd, state->buf + state->bytes_read, remaining);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break; 
            }
            closed = 1;
            break;
        }
        if (n == 0) {
            closed = 1; 
            break;
        }

        state->bytes_read += n;
        state->buf[state->bytes_read] = '\0';

        char *header_end = strstr(state->buf, "\r\n\r\n");
        if (header_end) {
            char *body = header_end + 4;
            
            char *cl_ptr = strstr(state->buf, "Content-Length:");
            int content_length = 0;
            if (cl_ptr) {
                content_length = atoi(cl_ptr + 15);
            }

            int body_read = state->bytes_read - (body - state->buf);
            if (body_read >= content_length) {
                handle_http_request(client_fd, state, body, content_length);

                int total_consumed = (body - state->buf) + content_length;
                if (state->bytes_read > total_consumed) {
                    int unread = state->bytes_read - total_consumed;
                    memmove(state->buf, state->buf + total_consumed, unread);
                    state->bytes_read = unread;
                } else {
                    state->bytes_read = 0;
                }
            }
        }
    }

    if (closed) {
        epoll_ctl(epoll_fd, EPOLL_CTL_DEL, client_fd, NULL);
        close(client_fd);
    }
}

// --- MAIN TCP SERVER ---
int main(int argc, char **argv) {
    printf("--- INITIALIZING FAST C AVX2 SERVER ---\n");

    load_normalization();
    load_mcc_risk();

    int fd = open("dataset.bin", O_RDONLY);
    if (fd < 0) {
        perror("Critical error: dataset.bin not found");
        return 1;
    }
    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        perror("fstat error");
        close(fd);
        return 1;
    }
    void *map = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap error");
        close(fd);
        return 1;
    }

    fast_path_table = (uint8_t *)map;
    num_clusters = *(uint32_t *)((char *)map + 524288);
    centroids = (int16_t *)((char *)map + 524288 + 4);
    cluster_offsets = (uint32_t *)((char *)centroids + num_clusters * 16 * sizeof(int16_t));
    cluster_counts = (uint32_t *)((char *)cluster_offsets + num_clusters * sizeof(uint32_t));
    cluster_radii = (float *)((char *)cluster_counts + num_clusters * sizeof(uint32_t));
    sorted_data = (int16_t *)((char *)cluster_radii + num_clusters * sizeof(float));

    long long expected_min_size = 524288 + 4 + num_clusters * 16 * sizeof(int16_t) + num_clusters * sizeof(uint32_t) + num_clusters * sizeof(uint32_t) + num_clusters * sizeof(float);
    if (sb.st_size < expected_min_size) {
        fprintf(stderr, "Critical error: dataset.bin size mismatch.\n");
        munmap(map, sb.st_size);
        close(fd);
        return 1;
    }

    long long num_vectors = (sb.st_size - expected_min_size) / (16 * sizeof(int16_t));
    printf("Loaded %d IVF clusters and %lld mmapped vectors.\n", num_clusters, num_vectors);

    printf("Prefaulting memory pages...\n");
    int16_t dummy_vec[16] = {0};
    int16_t dummy_labels[5] = {0};
    for (int i = 0; i < 1000; i++) {
        find_k_nearest(dummy_vec, dummy_labels);
    }
    printf("Warming up completed! Search engine is ready.\n");

    // 4. CRIAR SOCKET TCP E BIND
    int server_port = 8080;
    char *port_env = getenv("PORT");
    if (port_env) {
        server_port = atoi(port_env);
    }

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Erro ao criar socket");
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(server_port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Erro bind");
        return 1;
    }

    if (listen(server_fd, 4096) < 0) {
        perror("Listen error");
        return 1;
    }

    set_nonblocking(server_fd);
    printf("Server running on port %d...\n", server_port);

    int epoll_fd = epoll_create1(0);
    if (epoll_fd < 0) {
        perror("epoll_create1 error");
        return 1;
    }

    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = server_fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &ev) < 0) {
        perror("epoll_ctl error");
        return 1;
    }

    struct epoll_event events[MAX_EVENTS];
    char write_buf[8192];

    while (1) {
        int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        if (nfds < 0) {
            if (errno == EINTR) continue;
            perror("epoll_wait error");
            break;
        }

        for (int i = 0; i < nfds; i++) {
            int fd_event = events[i].data.fd;

            if (fd_event == server_fd) {
                accept_new_connections(epoll_fd, server_fd);
            } else {
                handle_client_data(epoll_fd, fd_event);
            }
        }
    }

    close(server_fd);
    munmap(map, sb.st_size);
    close(fd);
    return 0;
}
