#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"
#include "mityaeva_d_contrast_enhancement_histogram_stretching/mpi/include/ops_mpi.hpp"
#include "mityaeva_d_contrast_enhancement_histogram_stretching/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

class ContrastEnhancementRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kImageSize_ = 512;
  InType input_data_;

  void SetUp() override {
    int width = kImageSize_;
    int height = kImageSize_;
    int total_pixels = width * height;

    input_data_.clear();
    input_data_.reserve(2 + total_pixels);
    if (width > 255 || height > 255) {
      width = 255;
      height = 255;
      total_pixels = width * height;
    }

    input_data_.push_back(static_cast<uint8_t>(width));
    input_data_.push_back(static_cast<uint8_t>(height));

    for (int i = 0; i < total_pixels; ++i) {
      int y = i / width;
      int x = i % width;
      uint8_t pixel_value = static_cast<uint8_t>((x + y) % 256);
      input_data_.push_back(pixel_value);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    if (output_data.size() < 2) {
      return false;
    }

    int out_width = static_cast<int>(output_data[0]);
    int out_height = static_cast<int>(output_data[1]);

    int expected_pixels = out_width * out_height;
    size_t expected_size = static_cast<size_t>(expected_pixels) + 2;

    if (output_data.size() != expected_size) {
      return false;
    }

    for (size_t i = 2; i < output_data.size(); ++i) {
      uint8_t pixel = output_data[i];
      if (pixel > 255) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ContrastEnhancementRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ContrastEnhancementMPI, ContrastEnhancementSEQ>(
    PPC_SETTINGS_mityaeva_d_contrast_enhancement_histogram_stretching);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ContrastEnhancementRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ContrastEnhancementRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
