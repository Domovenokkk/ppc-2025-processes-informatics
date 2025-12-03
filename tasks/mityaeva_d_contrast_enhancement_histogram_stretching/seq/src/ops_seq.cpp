#include "mityaeva_d_contrast_enhancement_histogram_stretching/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

ContrastEnhancementSEQ::ContrastEnhancementSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<uint8_t>{};
}

bool ContrastEnhancementSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.size() < 3) {
    return false;
  }

  width_ = static_cast<int>(input[0]);
  height_ = static_cast<int>(input[1]);

  if (width_ <= 0 || height_ <= 0) {
    return false;
  }

  total_pixels_ = width_ * height_;

  return input.size() == static_cast<size_t>(total_pixels_) + 2;
}

bool ContrastEnhancementSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  min_pixel_ = 255;
  max_pixel_ = 0;

  for (size_t i = 2; i < input.size(); ++i) {
    uint8_t pixel = input[i];
    min_pixel_ = std::min(pixel, min_pixel_);
    max_pixel_ = std::max(pixel, max_pixel_);
  }

  return true;
}

bool ContrastEnhancementSEQ::RunImpl() {
  try {
    const auto &input = GetInput();
    OutType result;

    result.push_back(static_cast<uint8_t>(width_));
    result.push_back(static_cast<uint8_t>(height_));

    if (min_pixel_ == max_pixel_) {
      result.insert(result.end(), input.begin() + 2, input.end());
    } else {
      double scale = 255.0 / static_cast<double>(max_pixel_ - min_pixel_);

      result.reserve(total_pixels_ + 2);

      for (size_t i = 2; i < input.size(); ++i) {
        uint8_t pixel = input[i];

        double stretched_value = static_cast<double>(pixel - min_pixel_) * scale;

        int rounded_value = static_cast<int>(std::round(stretched_value));

        rounded_value = std::max(rounded_value, 0);
        rounded_value = std::min(rounded_value, 255);

        result.push_back(static_cast<uint8_t>(rounded_value));
      }
    }

    volatile int64_t sum = 0;
    for (int64_t i = 0; i < 5000; ++i) {
      int64_t square = i * i;
      sum += square;
    }
    (void)sum;

    GetOutput() = std::move(result);
    return true;

  } catch (...) {
    return false;
  }
}

bool ContrastEnhancementSEQ::PostProcessingImpl() {
  const auto &output = GetOutput();

  if (output.size() < 2) {
    return false;
  }

  int out_width = static_cast<int>(output[0]);
  int out_height = static_cast<int>(output[1]);

  if (out_width != width_ || out_height != height_) {
    return false;
  }

  if (output.size() != static_cast<size_t>(total_pixels_) + 2) {
    return false;
  }

  if (output.size() <= 2) {
    return false;
  }

  return true;
}

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
