#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"
#include "mityaeva_d_striped_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"
#include "mityaeva_d_striped_horizontal_matrix_vector/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace mityaeva_d_striped_horizontal_matrix_vector {

class StripedHorizontalMatrixVectorRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 100;  // Еще уменьшим для SEQ версии
  InType input_data_;

  void SetUp() override {
    int rows = kMatrixSize_;
    int cols = kMatrixSize_;

    input_data_.clear();
    input_data_.reserve(2 + (rows * cols) + cols);

    // Добавляем размеры матрицы
    input_data_.push_back(static_cast<double>(rows));
    input_data_.push_back(static_cast<double>(cols));

    // Заполняем матрицу единицами для простоты проверки
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        input_data_.push_back(1.0);
      }
    }

    // Заполняем вектор единицами
    for (int j = 0; j < cols; ++j) {
      input_data_.push_back(1.0);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для MPI: только процесс 0 имеет выходные данные
    // Для SEQ: всегда есть выходные данные

    // Если выход пустой (для MPI процессов кроме 0), это нормально
    if (output_data.empty()) {
      return true;
    }

    // Если есть данные, проверяем их
    if (output_data.size() != static_cast<size_t>(kMatrixSize_)) {
      return false;
    }

    // Для матрицы из единиц и вектора из единиц результат = cols
    double expected_value = static_cast<double>(kMatrixSize_);
    const double epsilon = 1e-6;

    for (const auto &val : output_data) {
      if (std::abs(val - expected_value) > epsilon) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(StripedHorizontalMatrixVectorRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, StripedHorizontalMatrixVectorMPI, StripedHorizontalMatrixVectorSEQ>(
        PPC_SETTINGS_mityaeva_d_striped_horizontal_matrix_vector);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = StripedHorizontalMatrixVectorRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, StripedHorizontalMatrixVectorRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
