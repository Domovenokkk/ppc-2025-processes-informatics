#include "rychkova_d_sobel_edge_detection/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rychkova_d_sobel_edge_detection/common/include/common.hpp"

namespace rychkova_d_sobel_edge_detection {

namespace {

inline std::uint8_t ClampToU8(int value) {
  if (value < 0) {
    return 0;
  }
  if (value > 255) {
    return 255;
  }
  return static_cast<std::uint8_t>(value);
}

}  // namespace

SobelEdgeDetectionSEQ::SobelEdgeDetectionSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool SobelEdgeDetectionSEQ::ValidationImpl() {
  const auto &in = GetInput();
  if (in.width == 0 || in.height == 0) {
    return false;
  }
  if (in.channels != 1 && in.channels != 3) {
    return false;
  }

  const std::size_t expected = in.width * in.height * in.channels;
  if (in.data.size() != expected) {
    return false;
  }

  const auto &out = GetOutput();
  return out.data.empty() && out.width == 0 && out.height == 0;
}

bool SobelEdgeDetectionSEQ::PreProcessingImpl() {
  const auto &in = GetInput();

  const std::size_t pixels = in.width * in.height;
  gray_.assign(pixels, 0);
  out_data_.assign(pixels, 0);

  if (in.channels == 1) {
    std::ranges::copy(in.data, gray_.begin());
  } else {
    for (std::size_t idx_px = 0; idx_px < pixels; ++idx_px) {
      const std::size_t base = (idx_px * 3U);
      const std::uint8_t r = in.data[base + 0U];
      const std::uint8_t g = in.data[base + 1U];
      const std::uint8_t b = in.data[base + 2U];
      const int y = (77 * r + 150 * g + 29 * b) >> 8;
      gray_[idx_px] = static_cast<std::uint8_t>(y);
    }
  }

  auto &out = GetOutput();
  out.width = in.width;
  out.height = in.height;
  out.channels = 1;
  out.data.clear();

  return true;
}

bool SobelEdgeDetectionSEQ::RunImpl() {
  const auto &in = GetInput();
  const std::size_t width = in.width;
  const std::size_t height = in.height;

  if (width == 0 || height == 0) {
    return false;
  }

  if (width < 3 || height < 3) {
    std::ranges::fill(out_data_, 0);
    return true;
  }

  auto idx = [width](std::size_t col, std::size_t row) { return (row * width) + col; };

  for (std::size_t row = 1; (row + 1U) < height; ++row) {
    for (std::size_t col = 1; (col + 1U) < width; ++col) {
      const int p00 = static_cast<int>(gray_[idx(col - 1U, row - 1U)]);
      const int p10 = static_cast<int>(gray_[idx(col, row - 1U)]);
      const int p20 = static_cast<int>(gray_[idx(col + 1U, row - 1U)]);

      const int p01 = static_cast<int>(gray_[idx(col - 1U, row)]);
      const int p21 = static_cast<int>(gray_[idx(col + 1U, row)]);

      const int p02 = static_cast<int>(gray_[idx(col - 1U, row + 1U)]);
      const int p12 = static_cast<int>(gray_[idx(col, row + 1U)]);
      const int p22 = static_cast<int>(gray_[idx(col + 1U, row + 1U)]);

      const int gx = (-p00 + p20) + (-2 * p01 + 2 * p21) + (-p02 + p22);
      const int gy = (-p00 - 2 * p10 - p20) + (p02 + 2 * p12 + p22);

      int mag = std::abs(gx) + std::abs(gy);
      mag /= 4;

      out_data_[idx(col, row)] = ClampToU8(mag);
    }
  }

  return true;
}

bool SobelEdgeDetectionSEQ::PostProcessingImpl() {
  auto &out = GetOutput();
  out.data = out_data_;
  return (out.data.size() == out.width * out.height * out.channels);
}

}  // namespace rychkova_d_sobel_edge_detection
