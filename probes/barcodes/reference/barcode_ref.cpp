// Reference side of the barcode probe, built against the service's own zxing-cpp.
//   barcode_ref generate <images dir> <out dir>  writes raw BGR test images (<name>_<W>x<H>.bgr)
//   barcode_ref decode <dir>                     prints one JSON line per image with ZXing results
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include <opencv2/opencv.hpp>

#include "BitMatrix.h"
#include "MultiFormatWriter.h"
#include "ReadBarcode.h"

namespace fs = std::filesystem;

static void save(const cv::Mat& bgr, const fs::path& dir, const std::string& name) {
  auto path = dir / (name + "_" + std::to_string(bgr.cols) + "x" + std::to_string(bgr.rows) + ".bgr");
  cv::Mat m = bgr.isContinuous() ? bgr : bgr.clone();
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(m.data), m.total() * 3);
}

static cv::Mat render(ZXing::BarcodeFormat format, const std::string& text) {
  auto matrix = ZXing::MultiFormatWriter(format).setMargin(0).encode(text, 0, 0);
  cv::Mat code(matrix.height(), matrix.width(), CV_8UC1);
  for (int y = 0; y < matrix.height(); ++y)
    for (int x = 0; x < matrix.width(); ++x)
      code.at<uchar>(y, x) = matrix.get(x, y) ? 0 : 255;
  bool linear = matrix.height() <= 1;
  cv::Mat scaled;
  if (linear)
    cv::resize(code, scaled, cv::Size(code.cols * 3, 120), 0, 0, cv::INTER_NEAREST);
  else
    cv::resize(code, scaled, cv::Size(code.cols * 6, code.rows * 6), 0, 0, cv::INTER_NEAREST);
  cv::Mat bordered;
  cv::copyMakeBorder(scaled, bordered, 40, 40, 40, 40, cv::BORDER_CONSTANT, cv::Scalar(255));
  cv::Mat bgr;
  cv::cvtColor(bordered, bgr, cv::COLOR_GRAY2BGR);
  return bgr;
}

static void generate(const fs::path& images, const fs::path& out) {
  fs::create_directories(out);
  const std::vector<std::pair<ZXing::BarcodeFormat, std::string>> codes = {
    {ZXing::BarcodeFormat::QRCode, "https://example.com/falprs?id=12345"},
    {ZXing::BarcodeFormat::DataMatrix, "FALPRS-DM-0001"},
    {ZXing::BarcodeFormat::Aztec, "AZTEC 0123456789"},
    {ZXing::BarcodeFormat::PDF417, "PDF417 sample text 2026"},
    {ZXing::BarcodeFormat::Code128, "FALPRS-128"},
    {ZXing::BarcodeFormat::Code39, "FALPRS39"},
    {ZXing::BarcodeFormat::Code93, "FALPRS93"},
    {ZXing::BarcodeFormat::Codabar, "A123456B"},
    {ZXing::BarcodeFormat::EAN13, "4006381333931"},
    {ZXing::BarcodeFormat::EAN8, "96385074"},
    {ZXing::BarcodeFormat::UPCA, "036000291452"},
    {ZXing::BarcodeFormat::UPCE, "04252614"},
    {ZXing::BarcodeFormat::ITF, "00012345678905"},
  };
  cv::Mat scene = cv::imread((images / "test_001.jpg").string(), cv::IMREAD_COLOR);
  std::mt19937 rng(42);
  for (const auto& [format, text] : codes) {
    std::string name = ZXing::ToString(format);
    cv::Mat img = render(format, text);
    save(img, out, name + "_plain");
    cv::Mat r;
    cv::rotate(img, r, cv::ROTATE_90_CLOCKWISE); save(r, out, name + "_rot90");
    cv::rotate(img, r, cv::ROTATE_180); save(r, out, name + "_rot180");
    cv::Mat m = cv::getRotationMatrix2D(cv::Point2f(img.cols / 2.0f, img.rows / 2.0f), 12, 1.0);
    cv::warpAffine(img, r, m, img.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255)); save(r, out, name + "_rot12");
    cv::GaussianBlur(img, r, cv::Size(5, 5), 1.5); save(r, out, name + "_blur");
    cv::Mat noise(img.size(), CV_16SC3);
    cv::randn(noise, 0, 25);
    cv::Mat noisy;
    img.convertTo(noisy, CV_16SC3);
    noisy += noise;
    noisy.convertTo(r, CV_8UC3); save(r, out, name + "_noise");
    std::vector<uchar> jpeg;
    cv::imencode(".jpg", img, jpeg, {cv::IMWRITE_JPEG_QUALITY, 40});
    save(cv::imdecode(jpeg, cv::IMREAD_COLOR), out, name + "_jpeg40");
    cv::resize(img, r, cv::Size(), 0.5, 0.5, cv::INTER_AREA); save(r, out, name + "_half");
    cv::Mat composed = scene.clone();
    cv::Mat small;
    cv::resize(img, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    small.copyTo(composed(cv::Rect(200, 300, small.cols, small.rows))); save(composed, out, name + "_scene");
  }
  for (const auto& entry : fs::directory_iterator(images))
    if (entry.path().extension() == ".jpg")
      save(cv::imread(entry.path().string(), cv::IMREAD_COLOR), out, "photo_" + entry.path().stem().string());
}

static std::string escape(const std::string& s) {
  std::string out;
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
    else if (c < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
    else out += static_cast<char>(c);
  }
  return out;
}

static void decode(const fs::path& dir) {
  // The options and the BGR-as-RGB image format of the FRS workflow
  auto options = ZXing::ReaderOptions().setFormats(ZXing::BarcodeFormat::Any).setTryHarder(true).setTryRotate(true);
  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(dir))
    if (entry.path().extension() == ".bgr") files.push_back(entry.path());
  std::sort(files.begin(), files.end());
  for (const auto& path : files) {
    auto stem = path.stem().string();
    auto size = stem.substr(stem.rfind('_') + 1);
    int w = std::stoi(size.substr(0, size.find('x'))), h = std::stoi(size.substr(size.find('x') + 1));
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), {});
    auto start = std::chrono::steady_clock::now();
    auto barcodes = ZXing::ReadBarcodes(ZXing::ImageView(data.data(), w, h, ZXing::ImageFormat::RGB), options);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"file\":\"" << path.filename().string() << "\",\"ms\":" << ms << ",\"results\":[";
    bool first = true;
    for (const auto& b : barcodes) {
      if (b.format() == ZXing::BarcodeFormat::None) continue;
      if (!first) std::cout << ",";
      first = false;
      std::cout << "{\"format\":\"" << ZXing::ToString(b.format()) << "\",\"text\":\"" << escape(b.text()) << "\",\"position\":[";
      for (int i = 0; i < 4; ++i) std::cout << (i ? "," : "") << b.position()[i].x << "," << b.position()[i].y;
      std::cout << "]}";
    }
    std::cout << "]}" << std::endl;
  }
}

int main(int argc, char** argv) {
  std::string mode = argv[1];
  if (mode == "generate") generate(argv[2], argv[3]);
  else decode(argv[2]);
}
