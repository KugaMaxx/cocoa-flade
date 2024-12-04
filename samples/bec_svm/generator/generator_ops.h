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

class Detector {
public:
  explicit Detector(const cv::Size &resolution, float_t min_area,
                      size_t candidate_num, float_t threshold)
      : height_(resolution.height), 
        width_(resolution.width),
        min_area_(min_area), 
        candidate_num_(candidate_num),
        threshold_(threshold) {}

  std::vector<std::vector<float_t>> detect(const py::array_t<int64_t> &array) {
    // project
    cv::Mat cnt_image = cv::Mat::zeros(width_, height_, CV_32F);
    cv::Mat bin_image = cv::Mat::zeros(width_, height_, CV_8UC1);
    for (size_t i = 0; i < array.request().shape[0]; i++) {
      cnt_image.at<uint8_t>(array.at(i, 1), array.at(i, 2)) += 1;
      bin_image.at<uint8_t>(array.at(i, 1), array.at(i, 2)) = 255;
    }

    // morphological process
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::Mat dilated_image;
    cv::dilate(bin_image, dilated_image, kernel);

    cv::Mat eroded_image;
    cv::erode(dilated_image, eroded_image, kernel);

    // find possible regions
    RegionSet region_set = findContoursRect(eroded_image);

    // selective search
    std::vector<std::vector<float_t>> results = selectiveBoundingBox(cnt_image, region_set);

    return results;
  }

private:
  struct RegionSet {
  private:
    size_t _ind = 0;
    size_t _size = 0;

  public:
    std::vector<cv::Rect> rect;
    std::vector<cv::Point2f> center;
    std::vector<float_t> radius;
    std::vector<uint32_t> rank;
    std::vector<uint32_t> label;

    size_t size() { return _size; }

    RegionSet(){};
    RegionSet(size_t length) : _size(length), _ind(0) {
      rect.resize(_size);
      center.resize(_size);
      radius.resize(_size);
      rank.resize(_size);
      label.resize(_size);
    }

    inline void push_back(const cv::Rect &rect_, const cv::Point2f center_,
                          const float radius_) {
      rect[_ind] = rect_;
      center[_ind] = center_;
      radius[_ind] = radius_;
      label[_ind] = _ind;
      _ind++;
    }

    inline int find(int i) { return (label[i] == i) ? i : find(label[i]); }

    inline void group(int i, int j) {
      int x = find(i), y = find(j);
      if (x != y) {
        if (rank[x] <= rank[y]) {
          label[x] = y;
        } else {
          label[y] = x;
        }
        if (rank[x] == rank[y]) {
          rank[y]++;
        }
      }
      return;
    }
  };

  RegionSet findContoursRect(cv::Mat bin_image) {
    // find contours
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin_image, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);

    // construct set of rectangle
    RegionSet region_set(contours.size());
    for (const auto &contour : contours) {
      // approximate polyline
      std::vector<cv::Point> contour_poly;
      cv::approxPolyDP(contour, contour_poly, 3, true);

      // get boundary rectangle
      cv::Rect bound_rect = cv::boundingRect(contour_poly);

      // search minimal enclosing circle
      float radius;
      cv::Point2f center;
      cv::minEnclosingCircle(contour_poly, center, radius);

      // emplace back
      region_set.push_back(bound_rect, center, radius);
    }

    return region_set;
  }

  std::vector<std::vector<float_t>> selectiveBoundingBox(cv::Mat cnt_image, RegionSet &region_set) {
    // group rectangles by calculate similarity
    for (size_t i = 0; i < region_set.size(); i++) {
      for (size_t j = i + 1; j < region_set.size(); j++) {
        if (calcSimilarity(cnt_image, region_set, i, j) >= threshold_) {
          region_set.group(i, j);
        }
      }
    }

    // merge rectangles within same group
    std::map<int32_t, cv::Rect> rects;
    for (size_t i = 0; i < region_set.size(); i++) {
      int k = region_set.find(i);
      if (!rects.count(k)) {
        rects[k] = region_set.rect[i];
        continue;
      }
      rects[k] |= region_set.rect[i];
    }

    // get candidate regions
    std::vector<std::pair<int32_t, cv::Rect>> rankedRect;
    for (size_t i = 0; i < rects.size(); i++) {
      if (rects[i].area() < min_area_)
        continue;
      rankedRect.push_back(std::make_pair(i, rects[i]));
    }
    std::sort(rankedRect.begin(), rankedRect.end(),
              [](auto &left, auto &right) {
                return left.second.area() > right.second.area();
              });

    // convert to lists of coordinates
    std::vector<std::vector<float_t>> result;
    for (size_t i = 0; i < rankedRect.size() && i < candidate_num_; i++) {
        auto rect = rankedRect[i].second;
        std::vector<float_t> vect = {
          (float) rect.y / width_, 
          (float) rect.x / height_, 
          (float) rect.height / width_,
          (float) rect.width / height_
        };
        result.push_back(vect);
    }

    return result;
  }

  inline float_t calcSimilarity(cv::Mat &cnt_image, RegionSet &region_set, int i, int j) {
    // box location
    float_t dist = cv::norm(region_set.center[i] - region_set.center[j]);
    float_t sumR = region_set.radius[i] + region_set.radius[j];
    float_t box_score = dist < sumR ? 1. : sumR / dist;

    // event rate
    const auto &rect1 = region_set.rect[i];
    const auto &rect2 = region_set.rect[j];

    auto count_rate = [&cnt_image](const cv::Rect& rect) {
        float_t count = 0;
        for (int x = rect.x; x < rect.x + rect.width; ++x) {
            for (int y = rect.y; y < rect.y + rect.height; ++y) {
                if (cnt_image.at<uint8_t>(y, x) != 0) {
                    count++;
                }
            }
        }
        return count;
    };

    float_t rate1 = count_rate(rect1) / rect1.area();
    float_t rate2 = count_rate(rect2) / rect2.area();
    float_t rate_score = std::fmin(rate1, rate2) / std::fmax(rate1, rate2);

    // results
    return 0.7 * box_score + 0.3 * rate_score;
  }

  size_t height_;
  size_t width_;
  size_t candidate_num_;
  float_t min_area_;
  float_t threshold_;
};
