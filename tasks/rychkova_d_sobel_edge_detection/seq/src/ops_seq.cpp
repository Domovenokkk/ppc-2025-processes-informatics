#include "rychkova_d_sobel_edge_detection/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "rychkova_d_sobel_edge_detection/common/include/common.hpp"

namespace rychkova_d_sobel_edge_detection {

namespace {

inline uint8_t ClampToU8(int v) {
  if (v < 0) {
    return 0;
  }
  if (v > 255) {
    return 255;
  }
  return static_cast<uint8_t>(v);
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
    std::copy(in.data.begin(), in.data.end(), gray_.begin());
  } else {
    // RGB -> grayscale
    for (std::size_t i = 0; i < pixels; ++i) {
      const uint8_t r = in.data[i * 3 + 0];
      const uint8_t g = in.data[i * 3 + 1];
      const uint8_t b = in.data[i * 3 + 2];
      const int y = (77 * r + 150 * g + 29 * b) >> 8;
      gray_[i] = static_cast<uint8_t>(y);
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
  const std::size_t w = in.width;
  const std::size_t h = in.height;

  if (w == 0 || h == 0) {
    return false;
  }

  if (w < 3 || h < 3) {
    std::fill(out_data_.begin(), out_data_.end(), 0);
    return true;
  }

  auto idx = [w](std::size_t x, std::size_t y) { return y * w + x; };

  for (std::size_t y = 1; y + 1 < h; ++y) {
    for (std::size_t x = 1; x + 1 < w; ++x) {
      const int p00 = static_cast<int>(gray_[idx(x - 1, y - 1)]);
      const int p10 = static_cast<int>(gray_[idx(x, y - 1)]);
      const int p20 = static_cast<int>(gray_[idx(x + 1, y - 1)]);

      const int p01 = static_cast<int>(gray_[idx(x - 1, y)]);
      const int p21 = static_cast<int>(gray_[idx(x + 1, y)]);

      const int p02 = static_cast<int>(gray_[idx(x - 1, y + 1)]);
      const int p12 = static_cast<int>(gray_[idx(x, y + 1)]);
      const int p22 = static_cast<int>(gray_[idx(x + 1, y + 1)]);

      const int gx = (-p00 + p20) + (-2 * p01 + 2 * p21) + (-p02 + p22);
      const int gy = (-p00 - 2 * p10 - p20) + (p02 + 2 * p12 + p22);

      int mag = std::abs(gx) + std::abs(gy);

      mag /= 4;

      out_data_[idx(x, y)] = ClampToU8(mag);
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
