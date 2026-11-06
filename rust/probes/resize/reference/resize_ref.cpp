// Reference side of the resize probe: resizes random BGR images with the OpenCV the C++ service links.
//   resize_ref <out dir> [random cases]
// Writes cases.txt ("<case> <linear|area> <src w> <src h> <dst w> <dst h>" per line) and, per case,
// <case>.in and <case>.out (raw BGR).
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct Case {
  std::string mode;
  int sw, sh, dw, dh;
};

static uint64_t seed = 12345;
static uint32_t next() {
  seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<uint32_t>(seed >> 33);
}

static void write(const fs::path& path, const cv::Mat& m) {
  std::ofstream f(path, std::ios::binary);
  for (int y = 0; y < m.rows; ++y)
    f.write(reinterpret_cast<const char*>(m.ptr(y)), static_cast<std::streamsize>(m.cols) * 3);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: resize_ref <out dir> [random cases]\n";
    return 1;
  }
  const fs::path out = argv[1];
  const int random_cases = argc > 2 ? std::stoi(argv[2]) : 500;
  fs::create_directories(out);

  // Sizes the services use (plate and vehicle crops, full frames), then random ones
  std::vector<Case> cases = {
    {"linear", 1920, 1080, 640, 360}, {"linear", 1920, 1082, 640, 361}, {"linear", 2048, 1365, 1024, 682},
    {"linear", 1280, 720, 317, 178}, {"linear", 40, 13, 160, 40}, {"linear", 87, 29, 224, 224},
    {"linear", 317, 211, 640, 427}, {"linear", 7, 5, 50, 41}, {"linear", 33, 97, 31, 330},
    {"linear", 1, 1, 5, 5}, {"linear", 5, 1, 3, 9}, {"linear", 100, 100, 100, 100},
    {"area", 1920, 1080, 320, 180}, {"area", 1920, 1080, 640, 360}, {"area", 1000, 700, 317, 211},
    {"area", 40, 13, 160, 40}, {"area", 317, 211, 112, 112}, {"area", 300, 200, 100, 100},
  };
  for (int i = 0; i < random_cases; ++i) {
    const auto size = [](uint32_t max) { return static_cast<int>(next() % max) + 1; };
    cases.push_back({i % 2 == 0 ? "linear" : "area", size(400), size(400), size(700), size(700)});
  }

  std::ofstream list(out / "cases.txt");
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto& c = cases[i];
    cv::Mat src(c.sh, c.sw, CV_8UC3);
    for (int y = 0; y < c.sh; ++y)
      for (int x = 0; x < c.sw * 3; ++x)
        src.ptr(y)[x] = static_cast<uint8_t>(next());
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(c.dw, c.dh), 0, 0, c.mode == "linear" ? cv::INTER_LINEAR : cv::INTER_AREA);
    write(out / (std::to_string(i) + ".in"), src);
    write(out / (std::to_string(i) + ".out"), dst);
    list << i << ' ' << c.mode << ' ' << c.sw << ' ' << c.sh << ' ' << c.dw << ' ' << c.dh << '\n';
  }
  std::cout << "OpenCV " << CV_VERSION << ", " << cases.size() << " cases\n";
  return 0;
}
