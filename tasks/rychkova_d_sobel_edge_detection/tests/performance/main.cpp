#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rychkova_d_sobel_edge_detection/common/include/common.hpp"
#include "rychkova_d_sobel_edge_detection/mpi/include/ops_mpi.hpp"
#include "rychkova_d_sobel_edge_detection/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rychkova_d_sobel_edge_detection {

class RychkovaDRunPerfTestsSobel : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr std::size_t kW = 2048;
  static constexpr std::size_t kH = 2048;
  static constexpr std::size_t kCh = 1;

  InType input_data_{};

 protected:
  void SetUp() override {
    input_data_.width = kW;
    input_data_.height = kH;
    input_data_.channels = kCh;
    input_data_.data.resize(kW * kH * kCh);

    for (std::size_t idx = 0; idx < input_data_.data.size(); ++idx) {
      input_data_.data[idx] = static_cast<std::uint8_t>((idx * 37 + 13) % 256);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int mpi_inited = 0;
    MPI_Initialized(&mpi_inited);

    if (mpi_inited != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if ((mpi_inited != 0) && (rank != 0)) {
      return true;
    }

    if (output_data.width != input_data_.width) {
      return false;
    }
    if (output_data.height != input_data_.height) {
      return false;
    }
    if (output_data.channels != 1) {
      return false;
    }
    if (output_data.data.size() != input_data_.width * input_data_.height) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RychkovaDRunPerfTestsSobel, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SobelEdgeDetectionMPI, SobelEdgeDetectionSEQ>(
    PPC_SETTINGS_rychkova_d_sobel_edge_detection);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RychkovaDRunPerfTestsSobel::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RychkovaDRunPerfTestsSobel, kGtestValues, kPerfTestName);

}  // namespace rychkova_d_sobel_edge_detection
