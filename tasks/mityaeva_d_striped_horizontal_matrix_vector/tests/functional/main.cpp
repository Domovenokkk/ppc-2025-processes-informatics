#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"
#include "mityaeva_d_striped_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"
#include "mityaeva_d_striped_horizontal_matrix_vector/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace mityaeva_d_striped_horizontal_matrix_vector {

class StripedHorizontalMatrixVectorRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_index = std::get<0>(params);
    switch (test_index) {
      case 1: {
        // Матрица 2x2 * вектор 2
        // [1, 2]   [3]   [1*3 + 2*4]   [11]
        // [3, 4] * [4] = [3*3 + 4*4] = [25]
        input_data_ = {2, 2, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0};
        expected_output_ = {11.0, 25.0};
        break;
      }
      case 2: {
        // Матрица 3x2 * вектор 2
        // [1, 2]   [1]   [1*1 + 2*2]   [5]
        // [3, 4] * [2] = [3*1 + 4*2] = [11]
        // [5, 6]         [5*1 + 6*2]   [17]
        input_data_ = {3, 2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0};
        expected_output_ = {5.0, 11.0, 17.0};
        break;
      }
      case 3: {
        // Матрица 1x3 * вектор 3
        // [1, 2, 3] * [4, 5, 6] = [1*4 + 2*5 + 3*6] = [32]
        input_data_ = {1, 3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        expected_output_ = {32.0};
        break;
      }
      case 4: {
        // Матрица 2x3 * вектор 3 с отрицательными числами
        // [1, -2, 3]   [2]   [1*2 + (-2)*1 + 3*0]   [0]
        // [0,  4, 5] * [1] = [0*2 + 4*1 + 5*0]   = [4]
        //              [0]
        input_data_ = {2, 3, 1.0, -2.0, 3.0, 0.0, 4.0, 5.0, 2.0, 1.0, 0.0};
        expected_output_ = {0.0, 4.0};
        break;
      }
      case 5: {
        // Матрица 3x1 * вектор 1
        // [1]   [2]   [2]
        // [3] * [2] = [6]
        // [5]         [10]
        input_data_ = {3, 1, 1.0, 3.0, 5.0, 2.0};
        expected_output_ = {2.0, 6.0, 10.0};
        break;
      }
      case 6: {
        // Матрица 4x2 * вектор 2
        // [1, 0]   [1]   [1]
        // [0, 1] * [2] = [2]
        // [2, 0]         [2]
        // [0, 2]         [4]
        input_data_ = {4, 2, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 2.0};
        expected_output_ = {1.0, 2.0, 2.0, 4.0};
        break;
      }
      default:
        throw std::runtime_error("Unknown test index: " + std::to_string(test_index));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > 1e-6) {
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
  OutType expected_output_;
};

namespace {

TEST_P(StripedHorizontalMatrixVectorRunFuncTests, MatrixVectorMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(1, "2x2_matrix"), std::make_tuple(2, "3x2_matrix"),
                                            std::make_tuple(3, "1x3_matrix"), std::make_tuple(4, "negative_numbers"),
                                            std::make_tuple(5, "3x1_matrix"), std::make_tuple(6, "4x2_matrix")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<StripedHorizontalMatrixVectorMPI, InType>(
                                               kTestParam, PPC_SETTINGS_mityaeva_d_striped_horizontal_matrix_vector),
                                           ppc::util::AddFuncTask<StripedHorizontalMatrixVectorSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_mityaeva_d_striped_horizontal_matrix_vector));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    StripedHorizontalMatrixVectorRunFuncTests::PrintFuncTestName<StripedHorizontalMatrixVectorRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixVectorTests, StripedHorizontalMatrixVectorRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
