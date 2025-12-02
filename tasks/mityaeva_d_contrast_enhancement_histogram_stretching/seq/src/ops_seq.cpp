#include "mityaeva_d_contrast_enhancement_histogram_stretching/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

  if (width_ > 10000 || height_ > 10000) {
    return false;
  }

  total_pixels_ = width_ * height_;

  return input.size() == static_cast<size_t>(total_pixels_) + 2;
}

bool ContrastEnhancementSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  min_pixel_ = MAX_PIXEL_VALUE;
  max_pixel_ = MIN_PIXEL_VALUE;

  for (size_t i = 2; i < input.size(); ++i) {
    uint8_t pixel = input[i];
    if (pixel < min_pixel_) {
      min_pixel_ = pixel;
    }
    if (pixel > max_pixel_) {
      max_pixel_ = pixel;
    }
  }

  return true;
}

bool ContrastEnhancementSEQ::RunImpl() {
  const auto &input = GetInput();

  try {
    std::vector<uint8_t> result;
    result.reserve(input.size());

    result.push_back(static_cast<uint8_t>(width_));
    result.push_back(static_cast<uint8_t>(height_));

    if (min_pixel_ == max_pixel_) {
      for (size_t i = 2; i < input.size(); ++i) {
        result.push_back(input[i]);
      }
      GetOutput() = std::move(result);
      return true;
    }

    double scale =
        static_cast<double>(MAX_PIXEL_VALUE - MIN_PIXEL_VALUE) / static_cast<double>(max_pixel_ - min_pixel_);

    for (size_t i = 2; i < input.size(); ++i) {
      uint8_t pixel = input[i];

      double new_value = static_cast<double>(pixel - min_pixel_) * scale;

      uint8_t enhanced_pixel = static_cast<uint8_t>(std::clamp(new_value + 0.5, 0.0, 255.0));

      result.push_back(enhanced_pixel);
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

  int out_width = static_cast<int>(output[0]);
  int out_height = static_cast<int>(output[1]);

  if (out_width != width_ || out_height != height_) {
    return false;
  }

  if (output.size() != static_cast<size_t>(total_pixels_) + 2) {
    return false;
  }

  for (size_t i = 2; i < output.size(); ++i) {
    uint8_t pixel = output[i];
    if (pixel < MIN_PIXEL_VALUE || pixel > MAX_PIXEL_VALUE) {
      return false;
    }
  }

  return true;
}

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
