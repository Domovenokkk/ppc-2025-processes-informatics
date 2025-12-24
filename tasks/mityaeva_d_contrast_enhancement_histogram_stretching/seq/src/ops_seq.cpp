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
    const uint8_t pixel = input[i];
    min_pixel_ = std::min(min_pixel_, pixel);
    max_pixel_ = std::max(max_pixel_, pixel);
  }

  return true;
}

bool ContrastEnhancementSEQ::RunImpl() {
  try {
    const auto &input = GetInput();
    OutType result;
    result.reserve(static_cast<size_t>(total_pixels_) + 2);

    result.push_back(static_cast<uint8_t>(width_));
    result.push_back(static_cast<uint8_t>(height_));

    if (min_pixel_ == max_pixel_) {
      result.insert(result.end(), input.begin() + 2, input.end());
    } else {
      const double scale = 255.0 / static_cast<double>(max_pixel_ - min_pixel_);

      for (size_t i = 2; i < input.size(); ++i) {
        const uint8_t pixel = input[i];

        const double stretched_value = static_cast<double>(pixel - min_pixel_) * scale;
        int rounded_value = static_cast<int>(std::lround(stretched_value));

        rounded_value = std::clamp(rounded_value, 0, 255);
        result.push_back(static_cast<uint8_t>(rounded_value));
      }
    }

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

  const int out_width = static_cast<int>(output[0]);
  const int out_height = static_cast<int>(output[1]);

  if (out_width != width_ || out_height != height_) {
    return false;
  }

  return output.size() == static_cast<size_t>(total_pixels_) + 2;
}

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
