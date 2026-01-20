#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "rychkova_d_image_smoothing/common/include/common.hpp"
#include "rychkova_d_image_smoothing/mpi/include/ops_mpi.hpp"
#include "rychkova_d_image_smoothing/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rychkova_d_image_smoothing {

class RychkovaDRunPerfTestsImageSmoothing : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr std::size_t kWidth = 1024;
  static constexpr std::size_t kHeight = 768;
  static constexpr std::size_t kChannels = 3;

  InType input_data{};

  void SetUp() override {
    input_data.width = kWidth;
    input_data.height = kHeight;
    input_data.channels = kChannels;
    input_data.data.resize(kWidth * kHeight * kChannels);

    for (std::size_t idx = 0; idx < input_data.data.size(); ++idx) {
      input_data.data[idx] = static_cast<std::uint8_t>(((idx * 37U) + 13U) % 256U);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.data.empty()) {
      return true;
    }

    if (output_data.width != input_data.width) {
      return false;
    }
    if (output_data.height != input_data.height) {
      return false;
    }
    if (output_data.channels != input_data.channels) {
      return false;
    }
    if (output_data.data.size() != input_data.data.size()) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(RychkovaDRunPerfTestsImageSmoothing, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ImageSmoothingMPI, ImageSmoothingSEQ>(PPC_SETTINGS_rychkova_d_image_smoothing);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RychkovaDRunPerfTestsImageSmoothing::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RychkovaDRunPerfTestsImageSmoothing, kGtestValues, kPerfTestName);

}  // namespace rychkova_d_image_smoothing
