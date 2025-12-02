#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

using InType = std::vector<uint8_t>;
using OutType = std::vector<uint8_t>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

constexpr uint8_t MIN_PIXEL_VALUE = 0;
constexpr uint8_t MAX_PIXEL_VALUE = 255;

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
