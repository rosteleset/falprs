// Reference side of the warp probe: straightens random quads of random BGR images with the OpenCV
// the C++ service links, as LPRS straightens plates.
//   warp_ref <out dir> [random cases]
// Writes cases.txt ("<case> <src w> <src h> <dst w> <dst h> <8 corner floats> <9 matrix doubles>" per
// line, floats and doubles as the integers of their bits) and, per case, <case>.in and <case>.out
// (raw BGR). The corners map onto (0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1) of the output.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct Case {
  int sw, sh, dw, dh;
  float corners[8];
};

static uint64_t seed = 12345;
static uint32_t next() {
  seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<uint32_t>(seed >> 33);
}
static float uniform(float from, float to) { return from + (to - from) * static_cast<float>(next() % 1000000) / 1e6f; }

static void write(const fs::path& path, const cv::Mat& m) {
  std::ofstream f(path, std::ios::binary);
  for (int y = 0; y < m.rows; ++y)
    f.write(reinterpret_cast<const char*>(m.ptr(y)), static_cast<std::streamsize>(m.cols) * 3);
}

static uint32_t bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
static uint64_t bits(double v) { uint64_t b; std::memcpy(&b, &v, 8); return b; }

/// A plate-like quad: a rotated, sheared rectangle inside or across the image border.
static Case random_case(int sw, int sh, int dw, int dh, float jitter) {
  Case c{sw, sh, dw, dh, {}};
  const float x = uniform(-20, sw - 20), y = uniform(-10, sh - 10), w = uniform(20, sw / 2.f + 20), h = uniform(5, sh / 4.f + 5);
  const float base[8] = {x, y, x + w, y + h * 0.1f, x + w * 1.02f, y + h, x - w * 0.01f, y + h * 0.95f};
  for (int i = 0; i < 8; ++i)
    c.corners[i] = base[i] + uniform(-jitter, jitter);
  return c;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: warp_ref <out dir> [random cases]\n";
    return 1;
  }
  const fs::path out = argv[1];
  const int random_cases = argc > 2 ? std::stoi(argv[2]) : 500;
  fs::create_directories(out);

  // LPRNet plate inputs (320 wide, one or two lines), outputs lower than OpenCV's 16-row blocks and
  // wider than its 64-column ones, then degenerate corners, then random ones
  std::vector<Case> cases = {
    {1920, 1080, 320, 68, {812.3f, 640.7f, 1003.9f, 652.1f, 1001.2f, 701.8f, 809.6f, 690.4f}},
    {1920, 1080, 320, 187, {400.5f, 300.25f, 520.75f, 310.5f, 518.0f, 380.0f, 398.0f, 372.5f}},
    {1280, 720, 320, 68, {-15.5f, 700.0f, 120.0f, 705.0f, 118.0f, 740.0f, -17.0f, 736.0f}},
    {640, 480, 300, 5, {100.0f, 100.0f, 400.0f, 120.0f, 398.0f, 125.0f, 99.0f, 104.0f}},
    {640, 480, 1000, 3, {10.0f, 10.0f, 630.0f, 30.0f, 629.0f, 33.0f, 9.0f, 13.0f}},
    {640, 480, 64, 64, {100.0f, 100.0f, 200.0f, 90.0f, 210.0f, 200.0f, 95.0f, 210.0f}},
    {640, 480, 65, 17, {100.0f, 100.0f, 200.0f, 90.0f, 210.0f, 200.0f, 95.0f, 210.0f}},
    {200, 100, 320, 68, {50.0f, 50.0f, 60.0f, 50.0f, 70.0f, 50.0f, 80.0f, 50.0f}},
    {200, 100, 320, 68, {50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f}},
  };
  for (int i = 0; i < random_cases; ++i) {
    const int sw = static_cast<int>(next() % 1900) + 20, sh = static_cast<int>(next() % 1060) + 20;
    const int dw = static_cast<int>(next() % 400) + 1, dh = static_cast<int>(next() % 200) + 1;
    cases.push_back(random_case(sw, sh, i % 3 == 0 ? 320 : dw, i % 3 == 0 ? (i % 2 ? 68 : 187) : dh, i % 7 == 0 ? 0.001f : 3.0f));
  }

  std::ofstream list(out / "cases.txt");
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto& c = cases[i];
    cv::Mat src(c.sh, c.sw, CV_8UC3);
    for (int y = 0; y < c.sh; ++y)
      for (int x = 0; x < c.sw * 3; ++x)
        src.ptr(y)[x] = static_cast<uint8_t>(next());
    const std::vector<cv::Point2f> from = {{c.corners[0], c.corners[1]}, {c.corners[2], c.corners[3]}, {c.corners[4], c.corners[5]}, {c.corners[6], c.corners[7]}};
    const std::vector<cv::Point2f> to = {{0, 0}, {static_cast<float>(c.dw - 1), 0}, {static_cast<float>(c.dw - 1), static_cast<float>(c.dh - 1)}, {0, static_cast<float>(c.dh - 1)}};
    const cv::Mat m = cv::getPerspectiveTransform(from, to);
    cv::Mat dst;
    cv::warpPerspective(src, dst, m, cv::Size(c.dw, c.dh));
    write(out / (std::to_string(i) + ".in"), src);
    write(out / (std::to_string(i) + ".out"), dst);
    list << i << ' ' << c.sw << ' ' << c.sh << ' ' << c.dw << ' ' << c.dh;
    for (float v : c.corners)
      list << ' ' << bits(v);
    for (int k = 0; k < 9; ++k)
      list << ' ' << bits(m.at<double>(k));
    list << '\n';
  }
  std::cout << "OpenCV " << CV_VERSION << ", " << cases.size() << " cases\n";
  return 0;
}
