// Reference side of the descriptor normalisation probe: fd / cv::norm(fd, NORM_L2), as the C++ FRS
// service normalises descriptors (frs_workflow.cpp, frs_caches.hpp), with the OpenCV it links.
//   normalize_ref <out dir> [random descriptors]
// Writes input.bin and output.bin: the descriptors before and after, 512 raw f32 each.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace fs = std::filesystem;

constexpr int DIM = 512;

static void append(std::ofstream& f, const cv::Mat& m) {
  f.write(reinterpret_cast<const char*>(m.ptr<float>()), DIM * sizeof(float));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: normalize_ref <out dir> [random descriptors]\n";
    return 1;
  }
  const fs::path dir = argv[1];
  const int count = argc > 2 ? std::stoi(argv[2]) : 100000;
  fs::create_directories(dir);
  std::ofstream input(dir / "input.bin", std::ios::binary), output(dir / "output.bin", std::ios::binary);

  std::mt19937 rng(20261109);
  std::normal_distribution<float> normal(0.f, 1.f);
  std::uniform_real_distribution<float> uniform(-1.f, 1.f);
  std::vector<cv::Mat> descriptors;
  // Special cases: zero (kept as is), one non-zero value, tiny and huge values
  descriptors.push_back(cv::Mat::zeros(1, DIM, CV_32F));
  cv::Mat one = cv::Mat::zeros(1, DIM, CV_32F);
  one.at<float>(7) = -3.f;
  descriptors.push_back(one);
  for (float scale : {1e-20f, 1e-6f, 1e6f, 1e18f}) {
    cv::Mat m(1, DIM, CV_32F);
    for (int i = 0; i < DIM; ++i) m.at<float>(i) = normal(rng) * scale;
    descriptors.push_back(m);
  }
  for (int n = 0; n < count; ++n) {
    cv::Mat m(1, DIM, CV_32F);
    // Model outputs have no fixed scale: spread them over several orders of magnitude
    const float scale = std::pow(10.f, static_cast<float>(n % 9) - 4.f);
    for (int i = 0; i < DIM; ++i) m.at<float>(i) = (n % 2 ? normal(rng) : uniform(rng)) * scale;
    descriptors.push_back(m);
  }

  for (const cv::Mat& fd : descriptors) {
    append(input, fd);
    // As frs_workflow.cpp
    double norm_l2 = cv::norm(fd, cv::NORM_L2);
    if (norm_l2 <= 0.0)
      norm_l2 = 1.0;
    cv::Mat normalized = fd / norm_l2;
    append(output, normalized);
  }
  std::cout << descriptors.size() << " descriptors written to " << dir << "\n";
  return 0;
}
