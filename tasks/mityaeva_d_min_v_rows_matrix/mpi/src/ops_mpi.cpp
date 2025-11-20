#include "mityaeva_d_min_v_rows_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "mityaeva_d_min_v_rows_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace mityaeva_d_min_v_rows_matrix {

MinValuesInRowsMPI::MinValuesInRowsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>{0};
}

bool MinValuesInRowsMPI::ValidationImpl() {
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

bool MinValuesInRowsMPI::PreProcessingImpl() {
  return true;
}

bool MinValuesInRowsMPI::RunImpl() {
  const auto &input = GetInput();

  try {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = input[0];
    int cols = input[1];

    int rows_per_process = rows / size;
    int remainder = rows % size;

    int my_rows = rows_per_process;
    if (rank < remainder) {
      my_rows++;
    }

    int start_row = 0;
    for (int i = 0; i < rank; ++i) {
      start_row += (rows_per_process + (i < remainder ? 1 : 0));
    }

    std::vector<int> local_result;
    local_result.reserve(my_rows);

    for (int i = 0; i < my_rows; ++i) {
      int global_row = start_row + i;
      int row_start_index = 2 + global_row * cols;

      if (cols == 0) {
        local_result.push_back(0);
        continue;
      }

      int min_val = input[row_start_index];
      for (int j = 1; j < cols; ++j) {
        if (input[row_start_index + j] < min_val) {
          min_val = input[row_start_index + j];
        }
      }
      local_result.push_back(min_val);
    }

    if (rank == 0) {
      std::vector<int> global_result;
      global_result.reserve(rows);

      global_result.insert(global_result.end(), local_result.begin(), local_result.end());

      for (int src = 1; src < size; ++src) {
        int src_rows = rows_per_process;
        if (src < remainder) {
          src_rows++;
        }

        if (src_rows > 0) {
          std::vector<int> recv_buffer(src_rows);
          MPI_Recv(recv_buffer.data(), src_rows, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          global_result.insert(global_result.end(), recv_buffer.begin(), recv_buffer.end());
        }
      }

      auto &output = GetOutput();
      output.clear();
      output.reserve(rows + 1);
      output.push_back(rows);
      output.insert(output.end(), global_result.begin(), global_result.end());

    } else {
      if (my_rows > 0) {
        MPI_Send(local_result.data(), my_rows, MPI_INT, 0, 0, MPI_COMM_WORLD);
      }
      auto &output = GetOutput();
      output.clear();
      output.push_back(0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return true;

  } catch (...) {
    return false;
  }
}

bool MinValuesInRowsMPI::PostProcessingImpl() {
  const auto &output = GetOutput();
  return !output.empty();
}

}  // namespace mityaeva_d_min_v_rows_matrix
