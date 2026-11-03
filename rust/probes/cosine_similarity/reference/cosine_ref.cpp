// Reference side of the cosine similarity probe: cosineSimilaritySIMD of frs_api.hpp (AVX) and
// cosineSimilarity of frs_workflow.cpp (OpenCV universal intrinsics), built with the service's
// flags (-O3 -mavx).
//   cosine_ref <pairs.bin> <values.bin>
// Prints nanoseconds per call (best of 3, memory-bound and cache-resident data) and writes the
// values as f64: all full cosines, then all dot products.
#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/core/hal/intrin.hpp>

constexpr int DIM = 512;
constexpr size_t PAIRS = 10000;
constexpr size_t CALLS = 1000000;
constexpr size_t HOT_PAIRS = 64;

// frs_api.hpp
static inline float reduceSum(const __m256& a) {
  __m256 s0 = _mm256_hadd_ps(a, a);
  s0 = _mm256_hadd_ps(s0, s0);
  auto s1 = _mm256_extractf128_ps(s0, 1);
  s1 = _mm_add_ps(_mm256_castps256_ps128(s0), s1);
  return _mm_cvtss_f32(s1);
}

static inline double cosineSimilaritySIMD(const float* d1, const float* d2) {
  constexpr int step = 8;
  __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps(), sum3 = _mm256_setzero_ps();
  __m256 sum0_sqr1 = _mm256_setzero_ps(), sum1_sqr1 = _mm256_setzero_ps(), sum2_sqr1 = _mm256_setzero_ps(),
         sum3_sqr1 = _mm256_setzero_ps();
  __m256 sum0_sqr2 = _mm256_setzero_ps(), sum1_sqr2 = _mm256_setzero_ps(), sum2_sqr2 = _mm256_setzero_ps(),
         sum3_sqr2 = _mm256_setzero_ps();
  for (int i = 0; i < DIM; i += 4 * step) {
    __m256 a0 = _mm256_loadu_ps(d1 + i + 0 * step);
    __m256 b0 = _mm256_loadu_ps(d2 + i + 0 * step);
    sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a0, b0));
    sum0_sqr1 = _mm256_add_ps(sum0_sqr1, _mm256_mul_ps(a0, a0));
    sum0_sqr2 = _mm256_add_ps(sum0_sqr2, _mm256_mul_ps(b0, b0));
    a0 = _mm256_loadu_ps(d1 + i + 1 * step);
    b0 = _mm256_loadu_ps(d2 + i + 1 * step);
    sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(a0, b0));
    sum1_sqr1 = _mm256_add_ps(sum1_sqr1, _mm256_mul_ps(a0, a0));
    sum1_sqr2 = _mm256_add_ps(sum1_sqr2, _mm256_mul_ps(b0, b0));
    a0 = _mm256_loadu_ps(d1 + i + 2 * step);
    b0 = _mm256_loadu_ps(d2 + i + 2 * step);
    sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(a0, b0));
    sum2_sqr1 = _mm256_add_ps(sum2_sqr1, _mm256_mul_ps(a0, a0));
    sum2_sqr2 = _mm256_add_ps(sum2_sqr2, _mm256_mul_ps(b0, b0));
    a0 = _mm256_loadu_ps(d1 + i + 3 * step);
    b0 = _mm256_loadu_ps(d2 + i + 3 * step);
    sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(a0, b0));
    sum3_sqr1 = _mm256_add_ps(sum3_sqr1, _mm256_mul_ps(a0, a0));
    sum3_sqr2 = _mm256_add_ps(sum3_sqr2, _mm256_mul_ps(b0, b0));
  }
  sum0 = _mm256_add_ps(sum0, sum1);
  sum2 = _mm256_add_ps(sum2, sum3);
  sum0 = _mm256_add_ps(sum0, sum2);
  sum0_sqr1 = _mm256_add_ps(sum0_sqr1, sum1_sqr1);
  sum2_sqr1 = _mm256_add_ps(sum2_sqr1, sum3_sqr1);
  sum0_sqr1 = _mm256_add_ps(sum0_sqr1, sum2_sqr1);
  sum0_sqr2 = _mm256_add_ps(sum0_sqr2, sum1_sqr2);
  sum2_sqr2 = _mm256_add_ps(sum2_sqr2, sum3_sqr2);
  sum0_sqr2 = _mm256_add_ps(sum0_sqr2, sum2_sqr2);
  return reduceSum(sum0) / sqrt(reduceSum(sum0_sqr1)) / sqrt(reduceSum(sum0_sqr2));
}

// frs_workflow.cpp, on pre-normalised descriptors
static inline double cosineSimilarity(const float* ptr1, const float* ptr2) {
  constexpr int step = cv::v_float32::nlanes;
  cv::v_float32 sum0 = cv::vx_setzero_f32(), sum1 = cv::vx_setzero_f32(), sum2 = cv::vx_setzero_f32(),
                sum3 = cv::vx_setzero_f32();
  for (int i = 0; i < DIM; i += 4 * step) {
    cv::v_float32 v1 = cv::vx_load(ptr1 + i + 0 * step);
    cv::v_float32 v2 = cv::vx_load(ptr2 + i + 0 * step);
    sum0 += v1 * v2;
    v1 = cv::vx_load(ptr1 + i + 1 * step);
    v2 = cv::vx_load(ptr2 + i + 1 * step);
    sum1 += v1 * v2;
    v1 = cv::vx_load(ptr1 + i + 2 * step);
    v2 = cv::vx_load(ptr2 + i + 2 * step);
    sum2 += v1 * v2;
    v1 = cv::vx_load(ptr1 + i + 3 * step);
    v2 = cv::vx_load(ptr2 + i + 3 * step);
    sum3 += v1 * v2;
  }
  sum0 += sum1 + sum2 + sum3;
  return v_reduce_sum(sum0);
}

template <typename F>
static double time(F f, const std::vector<float>& data, size_t pairs) {
  double best = 1e100;
  for (int run = 0; run < 3; ++run) {
    volatile double sink = 0.0;
    double sum = 0.0;
    auto start = std::chrono::steady_clock::now();
    size_t done = 0;
    while (done < CALLS)
      for (size_t i = 0; i < pairs && done < CALLS; ++i, ++done) {
        const float* a = data.data() + 2 * i * DIM;
        asm volatile("" : "+r"(a));  // keep the call from being hoisted out of the loop
        sum += f(a, a + DIM);
      }
    best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / CALLS);
    sink = sum;
  }
  return best;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: cosine_ref <pairs.bin> <values.bin>\n";
    return 1;
  }
  std::ifstream in(argv[1], std::ios::binary | std::ios::ate);
  std::vector<float> data(in.tellg() / sizeof(float));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
  if (data.size() != 2 * PAIRS * DIM) {
    std::cerr << "expected " << PAIRS << " pairs of " << DIM << " floats\n";
    return 1;
  }

  std::cout.precision(4);
  std::cout << "v_float32::nlanes " << cv::v_float32::nlanes << "\n";
  std::cout << "cosineSimilaritySIMD ns (memory) " << time(cosineSimilaritySIMD, data, PAIRS) << "  ns (cache) "
            << time(cosineSimilaritySIMD, data, HOT_PAIRS) << "\n";
  std::cout << "cosineSimilarity     ns (memory) " << time(cosineSimilarity, data, PAIRS) << "  ns (cache) "
            << time(cosineSimilarity, data, HOT_PAIRS) << "\n";

  std::vector<double> values;
  for (size_t i = 0; i < PAIRS; ++i) values.push_back(cosineSimilaritySIMD(data.data() + 2 * i * DIM, data.data() + (2 * i + 1) * DIM));
  for (size_t i = 0; i < PAIRS; ++i) values.push_back(cosineSimilarity(data.data() + 2 * i * DIM, data.data() + (2 * i + 1) * DIM));
  std::ofstream(argv[2], std::ios::binary).write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}
