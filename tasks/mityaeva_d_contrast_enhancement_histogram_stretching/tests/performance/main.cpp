#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"
#include "mityaeva_d_contrast_enhancement_histogram_stretching/mpi/include/ops_mpi.hpp"
#include "mityaeva_d_contrast_enhancement_histogram_stretching/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

class ContrastEnhancementRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kImageWidth_ = 255;
  const int kImageHeight_ = 255;

 public:
  void SetUp() override {
    BaseRunPerfTests::SetUp();

    int width = kImageWidth_;
    int height = kImageHeight_;
    int total_pixels = width * height;

    input_data_.clear();
    input_data_.reserve(2 + total_pixels);

    input_data_.push_back(static_cast<uint8_t>(width));
    input_data_.push_back(static_cast<uint8_t>(height));

    for (int i = 0; i < total_pixels; ++i) {
      uint8_t pixel_value = static_cast<uint8_t>(50 + (i % 151));
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

    if (out_width != kImageWidth_) {
      return false;
    }

    if (out_height != kImageHeight_) {
      return false;
    }

    size_t expected_size = static_cast<size_t>(kImageWidth_ * kImageHeight_) + 2;

    if (output_data.size() != expected_size) {
      return false;
    }
    for (size_t i = 2; i < output_data.size(); ++i) {
      if (output_data[i] > 255) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
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
