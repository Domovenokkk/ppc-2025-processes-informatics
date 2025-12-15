#include "rychkova_d_image_smoothing/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>

#include "rychkova_d_image_smoothing/common/include/common.hpp"

namespace rychkova_d_image_smoothing {

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

  if (in.data.size() != in.width * in.height * in.channels) {
    return false;
  }

  return true;
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

  const std::size_t w = in.width;
  const std::size_t h = in.height;
  const std::size_t ch = in.channels;

  for (std::size_t y = 0; y < h; y++) {
    for (std::size_t x = 0; x < w; x++) {
      for (std::size_t c = 0; c < ch; c++) {
        int sum = 0;

        for (int dy = -1; dy <= 1; dy++) {
          long long ny = static_cast<long long>(y) + dy;
          if (ny < 0) {
            ny = 0;
          }
          if (ny >= static_cast<long long>(h)) {
            ny = h - 1;
          }

          for (int dx = -1; dx <= 1; dx++) {
            long long nx = static_cast<long long>(x) + dx;
            if (nx < 0) {
              nx = 0;
            }
            if (nx >= static_cast<long long>(w)) {
              nx = w - 1;
            }

            std::size_t index = (static_cast<std::size_t>(ny) * w + static_cast<std::size_t>(nx)) * ch + c;

            sum += in.data[index];
          }
        }

        std::size_t out_index = (y * w + x) * ch + c;
        out.data[out_index] = static_cast<uint8_t>(sum / 9);
      }
    }
  }

  return true;
}

bool ImageSmoothingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace rychkova_d_image_smoothing
