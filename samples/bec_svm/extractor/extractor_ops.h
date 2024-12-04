#pragma once

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


namespace py = pybind11;

class Converter {
public:
  explicit Converter(const cv::Size &resolution)
      : height_(resolution.height), 
        width_(resolution.width) {}

  std::vector<double> process(
    const py::array_t<int64_t> &array, 
    const py::array_t<double> &box) {
    
    // obtain box range
    int64_t tl_x = box.at(0) * width_;
    int64_t tl_y = box.at(1) * height_;
    int64_t dr_x = (box.at(0) + box.at(2)) * width_;
    int64_t dr_y = (box.at(1) + box.at(3)) * height_;
    double box_w = box.at(2) * width_;
    double box_h = box.at(3) * height_;
    
    // convert to binary image
    const auto array_size = array.request().shape[0];
    cv::Mat image_1 = cv::Mat::zeros(width_, height_, CV_8UC1); // forward
    cv::Mat image_2 = cv::Mat::zeros(width_, height_, CV_8UC1); // backward
    
    size_t count = 0;
    for (size_t i = 0; i < array_size; i++) {
      // coordinate
      int64_t x = array.at(i, 1);
      int64_t y = array.at(i, 2);

      // whether out of range
      if ((x < tl_x || x > dr_x) || (y < tl_y || y > dr_y)) {
        continue;
      }

      if (i <= array_size / 2) {
        image_1.at<uint8_t>(x, y) = 255;
      } else {
        image_2.at<uint8_t>(x, y) = 255;
      }
      count++;
    }

    // check whether image is zero
    if (cv::countNonZero(image_1) == 0) {
      image_1 = image_2;
    } else if (cv::countNonZero(image_2) == 0) {
      image_2 = image_1;
    }

    // combine images
    cv::Mat image = image_1 | image_2;

    // contour
    const std::vector<cv::Point> contour = findMainContour(image);
    const double contour_area = cv::contourArea(contour);
    const double arc_length = cv::arcLength(contour, true) + 1;

    // // corners
    // std::vector<cv::Point2f> corners;
    // cv::goodFeaturesToTrack(image, corners, max_corners, 0.01, 10);

    const double const_corners = 50;
    std::vector<cv::KeyPoint> corners;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(50);
    detector->detect(image, corners);

    // backward
    const std::vector<cv::Point> contour_1 = findMainContour(image_1);
    const double area_1 = cv::contourArea(contour_1) + 1;
    cv::Moments M_1 = cv::moments(contour_1);
    cv::Point centroid_1(M_1.m10 / (M_1.m00 + std::numeric_limits<double>::epsilon()), 
                         M_1.m01 / (M_1.m00 + std::numeric_limits<double>::epsilon()));

    // forward
    const std::vector<cv::Point> contour_2 = findMainContour(image_2);
    const double area_2 = cv::contourArea(contour_2) + 1;
    cv::Moments M_2 = cv::moments(contour_2);
    cv::Point centroid_2(M_2.m10 / (M_2.m00 + std::numeric_limits<double>::epsilon()), 
                         M_2.m01 / (M_2.m00 + std::numeric_limits<double>::epsilon()));

    std::vector<double> results{
      (double) (count) / (box_w * box_h),
      (double) (box_w / box_h),
      (double) (contour_area / (box_w * box_h)),
      (double) (4 * M_PI * contour_area / (arc_length * arc_length)),
      (double) std::fabs(area_1 - area_2) * 2 / (area_1 + area_2),
      (double) cv::norm(centroid_1 - centroid_2),
      (double) (corners.size() / const_corners),
    };

    return results;
  }

private:
  std::vector<cv::Point> findMainContour(cv::Mat &image) {
    // find contours
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);

    // sort contours
    std::sort(contours.begin(), contours.end(), 
              [](std::vector<cv::Point> &contour_1, std::vector<cv::Point> &contour_2){
                double area1 = cv::contourArea(contour_1);
                double area2 = cv::contourArea(contour_2);
                return (area1 > area2);
              });

    return contours[0];
  }

  size_t height_;
  size_t width_;
};
