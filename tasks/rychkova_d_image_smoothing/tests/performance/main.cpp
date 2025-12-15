#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "rychkova_d_image_smoothing/common/include/common.hpp"
#include "rychkova_d_image_smoothing/mpi/include/ops_mpi.hpp"
#include "rychkova_d_image_smoothing/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rychkova_d_image_smoothing {

class RychkovaDRunPerfTestsImageSmoothing : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    const std::size_t w = 1024;
    const std::size_t h = 768;
    const std::size_t ch = 3;

    input_data_.width = w;
    input_data_.height = h;
    input_data_.channels = ch;
    input_data_.data.resize(w * h * ch);

    for (std::size_t i = 0; i < input_data_.data.size(); ++i) {
      input_data_.data[i] = static_cast<uint8_t>((i * 37u + 13u) % 256u);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.data.empty()) {
      return true;
    }

    if (output_data.width != input_data_.width) {
      return false;
    }
    if (output_data.height != input_data_.height) {
      return false;
    }
    if (output_data.channels != input_data_.channels) {
      return false;
    }
    if (output_data.data.size() != input_data_.data.size()) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
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
