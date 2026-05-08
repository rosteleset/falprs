#include "image_preprocessing.hpp"

#include "absl/strings/substitute.h"

cv::Mat prepareBlobForYOLO(const cv::Mat& image, const int32_t target_width, const int32_t target_height, cv::Point2f& shift, double& scale)
{
  // for test
  // static int count = 0;
  // cv::imwrite(absl::Substitute("before_$0x$1_$2.png", image.cols, image.rows, count), image);

  const auto r_w = target_width / (image.cols * 1.0);
  const auto r_h = target_height / (image.rows * 1.0);
  scale = fmin(r_w, r_h);
  const auto ww = static_cast<int>(lround(scale * image.cols));
  const auto hh = static_cast<int>(lround(scale * image.rows));
  shift.x = static_cast<float>(target_width - ww) / 2;
  shift.y = static_cast<float>(target_height - hh) / 2;

  const cv::Mat out(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::Mat roi = out(cv::Rect(static_cast<int>(shift.x), static_cast<int>(shift.y), ww, hh));
  cv::resize(image, roi, roi.size(), 0, 0, cv::INTER_LINEAR);

  // for test
  // cv::imwrite(absl::Substitute("after_$0x$1_$2.png", image.cols, image.rows, count++), out);

  return cv::dnn::blobFromImage(
    out,
    1.0 / 255.0,          // normalization
    cv::Size(),           // size is already target
    cv::Scalar(0, 0, 0),  // without subtracting the mean
    true,                 // swapRB: BGR → RGB
    false                 // without crop
  );
}

cv::Mat prepareBlobForVcNet(const cv::Mat& img, const int32_t target_width, const int32_t target_height)
{
  cv::Mat out;
  cv::resize(img, out, cv::Size(target_width, target_height), 0, 0, cv::INTER_AREA);

  return cv::dnn::blobFromImage(
    out,
    1.0 / 127.5,                      // scale factor
    cv::Size(),                       // size is already target
    cv::Scalar(127.5, 127.5, 127.5),  // mean (subtracted before scaling)
    true,                             // swapRB: BGR → RGB
    false                             // without crop
  );
}

cv::Mat prepareBlobForScrfd(const cv::Mat& image, const int32_t target_width, const int32_t target_height, float& scale)
{
  const auto r_w = target_width / (image.cols * 1.0);
  const auto r_h = target_height / (image.rows * 1.0);
  int w, h;
  if (r_h > r_w)
  {
    w = target_width;
    h = static_cast<int>(r_w * image.rows);
  } else
  {
    w = static_cast<int>(r_h * image.cols);
    h = target_height;
  }
  scale = static_cast<float>(h) / static_cast<float>(image.rows);

  const cv::Mat out(target_height, target_width, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat roi = out(cv::Rect(0, 0, w, h));
  cv::resize(image, roi, roi.size(), 0, 0, cv::INTER_LINEAR);

  return cv::dnn::blobFromImage(
    out,
    1.0 / 128.0,                      // scale factor
    cv::Size(),                       // size is already target
    cv::Scalar(127.5, 127.5, 127.5),  // mean (subtracted before scaling)
    true,                             // swapRB: BGR → RGB
    false                             // without crop
  );
}

cv::Mat prepareBlobForGenet(const cv::Mat& img, const int32_t target_width, const int32_t target_height)
{
  cv::Mat out;
  cv::resize(img, out, cv::Size(target_width, target_height), 0, 0, cv::INTER_AREA);

  constexpr int32_t channels = 3;
  const int32_t plane_size = target_width * target_height;
  cv::Mat blob(1, channels * plane_size, CV_32F);
  auto* data = blob.ptr<float>();

  float* plane_r = data;
  float* plane_g = data + plane_size;
  float* plane_b = data + 2 * plane_size;

  // ImageNet normalization: (pixel / 255.0 - mean) / std, per channel (RGB)
  constexpr float inv255 = 1.0f / 255.0f;
  constexpr float mean_r = 0.485f, std_r = 0.229f;
  constexpr float mean_g = 0.456f, std_g = 0.224f;
  constexpr float mean_b = 0.406f, std_b = 0.225f;
  constexpr float inv_std_r = 1.0f / std_r;
  constexpr float inv_std_g = 1.0f / std_g;
  constexpr float inv_std_b = 1.0f / std_b;

  for (int h = 0; h < target_height; ++h)
  {
    const auto* row = out.ptr<uint8_t>(h);
    const int offset = h * target_width;
    for (int w = 0; w < target_width; ++w)
    {
      const int idx = offset + w;
      const int px = w * 3;
      plane_r[idx] = (static_cast<float>(row[px + 2]) * inv255 - mean_r) * inv_std_r;
      plane_g[idx] = (static_cast<float>(row[px + 1]) * inv255 - mean_g) * inv_std_g;
      plane_b[idx] = (static_cast<float>(row[px])     * inv255 - mean_b) * inv_std_b;
    }
  }

  return blob.reshape(1, {1, channels, target_height, target_width});
}

cv::Mat prepareBlobForArcface(const cv::Mat& img, const int32_t target_width, const int32_t target_height)
{
  cv::Mat out;
  cv::resize(img, out, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);

  return cv::dnn::blobFromImage(
    out,
    1.0 / 127.5,                      // scale factor
    cv::Size(),                       // size is already target
    cv::Scalar(127.5, 127.5, 127.5),  // mean (subtracted before scaling)
    true,                             // swapRB: BGR → RGB
    false                             // without crop
  );
}

// Letterbox for ViT
cv::Mat prepareBlobForLpcNet(const cv::Mat& img, const int32_t target_width, const int32_t target_height)
{
  const auto r_w = target_width / (img.cols * 1.0);
  const auto r_h = target_height / (img.rows * 1.0);
  const auto scale = fmin(r_w, r_h);
  const auto ww = static_cast<int>(lround(scale * img.cols));
  const auto hh = static_cast<int>(lround(scale * img.rows));
  const auto shift_x = static_cast<float>(target_width - ww) / 2;
  const auto shift_y = static_cast<float>(target_height - hh) / 2;

  const cv::Mat out(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::Mat roi = out(cv::Rect(static_cast<int>(shift_x), static_cast<int>(shift_y), ww, hh));
  cv::resize(img, roi, roi.size(), 0, 0, cv::INTER_LINEAR);

  return cv::dnn::blobFromImage(
    out,
    1.0 / 127.5,                      // scale factor
    cv::Size(),                       // size is already target
    cv::Scalar(127.5, 127.5, 127.5),  // mean (subtracted before scaling)
    true,                             // swapRB: BGR → RGB
    false                             // without crop
  );
}
