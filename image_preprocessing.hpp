#pragma once

#include <opencv2/opencv.hpp>

cv::Mat prepareBlobForYOLO(const cv::Mat& image, int32_t target_width, int32_t target_height, cv::Point2f& shift, double& scale);
cv::Mat prepareBlobForVcNet(const cv::Mat& img, int32_t target_width, int32_t target_height);
cv::Mat prepareBlobForScrfd(const cv::Mat& image, int32_t target_width, int32_t target_height, float& scale);
cv::Mat prepareBlobForGenet(const cv::Mat& img, int32_t target_width, int32_t target_height);
cv::Mat prepareBlobForArcface(const cv::Mat& img, int32_t target_width, int32_t target_height);
cv::Mat prepareBlobForLpcNet(const cv::Mat& img, int32_t target_width, int32_t target_height);

