#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace rychkova_d_image_smoothing {

struct Image {
  std::vector<uint8_t> data;
  std::size_t width = 0;
  std::size_t height = 0;
  std::size_t channels = 1;
};

using InType = Image;
using OutType = Image;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace rychkova_d_image_smoothing
