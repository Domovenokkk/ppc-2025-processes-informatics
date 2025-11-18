#include "rychkova_d_sum_matrix_columns/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "rychkova_d_sum_matrix_columns/common/include/common.hpp"

namespace rychkova_d_sum_matrix_columns {

RychkovaDSumMatrixColumnsMPI::RychkovaDSumMatrixColumnsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool RychkovaDSumMatrixColumnsMPI::ValidationImpl() {
  const auto &input = GetInput();

  if (input.empty()) {
    return true;
  }

  size_t cols = input[0].size();
  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool RychkovaDSumMatrixColumnsMPI::PreProcessingImpl() {
  const auto &input = GetInput();

  if (input.empty()) {
    GetOutput() = std::vector<int>{};
    return true;
  }

  GetOutput() = std::vector<int>(input[0].size(), 0);
  return true;
}

bool RychkovaDSumMatrixColumnsMPI::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  if (input.empty()) {
    return true;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t num_rows = input.size();
  size_t num_cols = input[0].size();

  size_t rows_per_process = num_rows / size;
  size_t remainder = num_rows % size;
  size_t start_row = (rank * rows_per_process) + std::min(static_cast<size_t>(rank), remainder);
  size_t end_row = start_row + rows_per_process + (std::cmp_less(rank, remainder) ? 1 : 0);

  std::vector<int> local_sums(num_cols, 0);

  for (size_t i = start_row; i < end_row; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      local_sums[j] += input[i][j];
    }
  }

  MPI_Allreduce(local_sums.data(), output.data(), static_cast<int>(num_cols), MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  return true;
}

bool RychkovaDSumMatrixColumnsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rychkova_d_sum_matrix_columns
