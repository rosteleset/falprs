// Reference side of the face alignment probe: aligns faces exactly as the FRS workflow does.
//   align_ref <bgr dir> <landmarks file> <out dir>
// Landmarks file lines: "<image file> <face index> x1 y1 ... x5 y5". Writes <image>_<face>.aligned
// (112x112 raw BGR) and prints "<image> <face> m00 m01 m02 m10 m11 m12" per face.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// alignFaceAffineTransform of frs_workflow.cpp
static cv::Mat align(const cv::Mat& frame, const cv::Mat& src, int face_width, int face_height, cv::Mat& mm) {
  const cv::Mat dst = (cv::Mat_<double>(5, 2) << 38.2946 / 112.0 * face_width, 51.6963 / 112.0 * face_height,
    73.5318 / 112.0 * face_width, 51.5014 / 112.0 * face_height,
    56.0252 / 112.0 * face_width, 71.7366 / 112.0 * face_height,
    41.5493 / 112.0 * face_width, 92.3655 / 112.0 * face_height,
    70.7299 / 112.0 * face_width, 92.2041 / 112.0 * face_height);
  cv::Mat inliers;
  mm = cv::estimateAffinePartial2D(src, dst, inliers, cv::LMEDS);
  cv::Mat r;
  warpAffine(frame, r, mm, cv::Size(face_width, face_height));
  return r;
}

int main(int argc, char** argv) {
  fs::path dir = argv[1], out = argv[3];
  fs::create_directories(out);
  std::ifstream in(argv[2]);
  std::string line;
  std::cout.precision(17);
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string file;
    int face;
    float landmark[10];
    ls >> file >> face;
    for (float& v : landmark) ls >> v;
    auto stem = fs::path(file).stem().string();
    auto size = stem.substr(stem.rfind('_') + 1);
    int w = std::stoi(size.substr(0, size.find('x'))), h = std::stoi(size.substr(size.find('x') + 1));
    std::ifstream img_in(dir / file, std::ios::binary);
    std::vector<uchar> data((std::istreambuf_iterator<char>(img_in)), {});
    cv::Mat frame(h, w, CV_8UC3, data.data());
    cv::Mat mm;
    cv::Mat aligned = align(frame, cv::Mat(5, 2, CV_32F, landmark), 112, 112, mm);
    auto name = out / (stem + "_" + std::to_string(face) + ".aligned");
    std::ofstream(name, std::ios::binary).write(reinterpret_cast<const char*>(aligned.data), aligned.total() * 3);
    std::cout << file << " " << face;
    for (int r = 0; r < 2; ++r)
      for (int c = 0; c < 3; ++c) std::cout << " " << mm.at<double>(r, c);
    std::cout << "\n";
  }
}
