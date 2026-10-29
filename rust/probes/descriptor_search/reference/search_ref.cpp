// Reference side of the descriptor search probe: the FRS cosineSimilarity (OpenCV universal
// intrinsics), built with the service's flags (-O3 -mavx).
//   search_ref <descriptors.bin> <queries.bin> <dimension>
// Prints the best of 3 timings and, per query, the index and similarity of the best match.
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/core/hal/intrin.hpp>

// cosineSimilarity of frs_workflow.cpp on pre-normalised descriptors
static double cosineSimilarity(const float* ptr1, const float* ptr2, int cols) {
  constexpr int step = cv::v_float32::nlanes;
  cv::v_float32 sum0 = cv::vx_setzero_f32();
  cv::v_float32 sum1 = cv::vx_setzero_f32();
  cv::v_float32 sum2 = cv::vx_setzero_f32();
  cv::v_float32 sum3 = cv::vx_setzero_f32();
  for (int i = 0; i < cols; i += 4 * step) {
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

static std::vector<float> read(const char* path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  std::vector<float> data(in.tellg() / sizeof(float));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
  return data;
}

int main(int argc, char** argv) {
  auto db = read(argv[1]), queries = read(argv[2]);
  int dim = std::atoi(argv[3]);
  size_t n = db.size() / dim, q = queries.size() / dim;
  std::vector<size_t> best_index(q);
  std::vector<double> best_value(q);
  double best_ms = 1e100;
  for (int run = 0; run < 3; ++run) {
    auto start = std::chrono::steady_clock::now();
    for (size_t j = 0; j < q; ++j) {
      double max_value = -1.0;
      size_t max_index = 0;
      for (size_t i = 0; i < n; ++i)
        if (double v = cosineSimilarity(queries.data() + j * dim, db.data() + i * dim, dim); v > max_value) {
          max_value = v;
          max_index = i;
        }
      best_index[j] = max_index;
      best_value[j] = max_value;
    }
    best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
  }
  std::cout.precision(9);
  std::cout << "ms " << best_ms << "\n";
  for (size_t j = 0; j < q; ++j) std::cout << best_index[j] << " " << best_value[j] << "\n";
}
