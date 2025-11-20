#include "mityaeva_d_min_v_rows_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "mityaeva_d_min_v_rows_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace mityaeva_d_min_v_rows_matrix {

MinValuesInRowsSEQ::MinValuesInRowsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>{0};
}

bool MinValuesInRowsSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.empty() || input.size() < 2) {
    return false;
  }

  int rows = input[0];
  int cols = input[1];

  if (rows <= 0 || cols <= 0) {
    return false;
  }

  if (input.size() != static_cast<size_t>(2 + rows * cols)) {
    return false;
  }

  return true;
}

bool MinValuesInRowsSEQ::PreProcessingImpl() {
  return true;
}

bool MinValuesInRowsSEQ::RunImpl() {
  const auto &input = GetInput();

  try {
    int rows = input[0];
    int cols = input[1];

    std::vector<int> result;
    result.reserve(rows);

    size_t data_index = 2;

    for (int i = 0; i < rows; ++i) {
      if (cols == 0) {
        result.push_back(0);
        continue;
      }

      int min_val = input[data_index];

      for (int j = 1; j < cols; ++j) {
        if (input[data_index + j] < min_val) {
          min_val = input[data_index + j];
        }
      }

      result.push_back(min_val);
      data_index += cols;
    }

    auto &output = GetOutput();
    output.clear();
    output.reserve(rows + 1);
    output.push_back(static_cast<int>(result.size()));

    for (int val : result) {
      output.push_back(val);
    }

    return true;

  } catch (...) {
    return false;
  }
}

bool MinValuesInRowsSEQ::PostProcessingImpl() {
  const auto &output = GetOutput();
  return !output.empty() && output[0] > 0;
}

}  // namespace mityaeva_d_min_v_rows_matrix
