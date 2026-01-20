#include "rychkova_d_image_smoothing/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "rychkova_d_image_smoothing/common/include/common.hpp"

namespace rychkova_d_image_smoothing {

namespace {

inline std::size_t ClampCoord(std::int64_t v, std::size_t max_val) {
  return static_cast<std::size_t>(std::clamp<std::int64_t>(v, 0, static_cast<std::int64_t>(max_val) - 1));
}

}  // namespace

ImageSmoothingSEQ::ImageSmoothingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool ImageSmoothingSEQ::ValidationImpl() {
  const auto &in = GetInput();
  if (in.width == 0 || in.height == 0) {
    return false;
  }
  if (in.channels != 1 && in.channels != 3) {
    return false;
  }
  return in.data.size() == in.width * in.height * in.channels;
}

bool ImageSmoothingSEQ::PreProcessingImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  out.width = in.width;
  out.height = in.height;
  out.channels = in.channels;
  out.data.assign(in.data.size(), 0);

  return true;
}

bool ImageSmoothingSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  const std::size_t width = in.width;
  const std::size_t height = in.height;
  const std::size_t channels = in.channels;

  for (std::size_t yy = 0; yy < height; ++yy) {
    for (std::size_t xx = 0; xx < width; ++xx) {
      for (std::size_t cc = 0; cc < channels; ++cc) {
        int sum = 0;

        for (int dy = -1; dy <= 1; ++dy) {
          const auto ny = ClampCoord(static_cast<std::int64_t>(yy) + dy, height);

          for (int dx = -1; dx <= 1; ++dx) {
            const auto nx = ClampCoord(static_cast<std::int64_t>(xx) + dx, width);

            const auto idx = (((ny * width) + nx) * channels) + cc;

            sum += static_cast<int>(in.data[idx]);
          }
        }

        const auto out_idx = (((yy * width) + xx) * channels) + cc;
        out.data[out_idx] = static_cast<std::uint8_t>(sum / 9);
      }
    }
  }

  return true;
}

bool ImageSmoothingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace rychkova_d_image_smoothing
