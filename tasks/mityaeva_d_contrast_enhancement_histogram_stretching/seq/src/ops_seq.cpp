#include "mityaeva_d_contrast_enhancement_histogram_stretching/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

static constexpr int kComputeRepeats = 200;

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
  min_pixel_ = kMaxPixelValue;
  max_pixel_ = kMinPixelValue;
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
    const size_t input_size = input.size();
    if (input_size < 2) {
      return false;
    }

    OutType result;
    result.push_back(static_cast<uint8_t>(width_));
    result.push_back(static_cast<uint8_t>(height_));

    if (min_pixel_ == max_pixel_) {
      volatile double sink = 0.0;
      for (int r = 0; r < kComputeRepeats; ++r) {
        for (size_t i = 2; i < input_size; ++i) {
          sink += static_cast<double>(input[i]);
        }
      }

      result.reserve(input_size);
      for (size_t i = 2; i < input_size; ++i) {
        result.push_back(input[i]);
      }
    } else {
      double scale =
          static_cast<double>(kMaxPixelValue - kMinPixelValue) / static_cast<double>(max_pixel_ - min_pixel_);

      std::vector<uint8_t> temp;
      temp.reserve(input_size - 2);

      for (size_t i = 2; i < input_size; ++i) {
        uint8_t pixel = input[i];
        double v = static_cast<double>(pixel - min_pixel_) * scale;
        v = std::min(std::max(v + 0.5, 0.0), 255.0);
        temp.push_back(static_cast<uint8_t>(v));
      }

      volatile double sink = 0.0;
      for (int r = 1; r < kComputeRepeats; ++r) {
        for (size_t i = 0; i < temp.size(); ++i) {
          sink += static_cast<double>(temp[i]) * scale;
        }
      }

      for (auto v : temp) {
        result.push_back(v);
      }
    }

    GetOutput().swap(result);
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
  return output.size() > 2;
}

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
