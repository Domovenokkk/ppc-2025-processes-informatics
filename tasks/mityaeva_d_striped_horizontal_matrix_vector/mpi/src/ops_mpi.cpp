#include "mityaeva_d_striped_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"

namespace mityaeva_d_striped_horizontal_matrix_vector {

StripedHorizontalMatrixVectorMPI::StripedHorizontalMatrixVectorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>{0.0};
}

bool StripedHorizontalMatrixVectorMPI::ValidationImpl() {
  const auto &input = GetInput();

  if (input.empty() || input.size() < 3) {
    return false;
  }

  int rows = static_cast<int>(input[0]);
  int cols = static_cast<int>(input[1]);

  if (rows <= 0 || cols <= 0) {
    return false;
  }

  if (static_cast<int>(input[2]) != cols) {
    return false;
  }

  size_t expected_size = 3 + (static_cast<size_t>(rows) * static_cast<size_t>(cols)) + static_cast<size_t>(cols);

  return input.size() == expected_size;
}

bool StripedHorizontalMatrixVectorMPI::PreProcessingImpl() {
  return true;
}

std::vector<double> ProcessLocalRows(const std::vector<double> &input, int start_row, int my_rows, int cols,
                                     int total_rows) {
  std::vector<double> local_result;
  local_result.reserve(my_rows);

  size_t matrix_start_idx = 3;
  size_t vector_start_idx = matrix_start_idx + (static_cast<size_t>(total_rows) * static_cast<size_t>(cols));

  for (int i = 0; i < my_rows; ++i) {
    int global_row = start_row + i;
    double sum = 0.0;

    size_t row_start = matrix_start_idx + (static_cast<size_t>(global_row) * static_cast<size_t>(cols));

    for (int j = 0; j < cols; ++j) {
      double matrix_element = input[row_start + j];
      double vector_element = input[vector_start_idx + j];
      sum += matrix_element * vector_element;
    }

    local_result.push_back(sum);
  }

  return local_result;
}

void GatherResults(int rank, int size, int rows, int rows_per_process, int remainder,
                   const std::vector<double> &local_result, std::vector<double> &output) {
  if (rank == 0) {
    std::vector<double> global_result;
    global_result.reserve(rows);

    global_result.insert(global_result.end(), local_result.begin(), local_result.end());

    for (int src = 1; src < size; ++src) {
      int src_rows = rows_per_process;
      if (src < remainder) {
        src_rows++;
      }

      if (src_rows > 0) {
        std::vector<double> recv_buffer(src_rows);
        MPI_Recv(recv_buffer.data(), src_rows, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        global_result.insert(global_result.end(), recv_buffer.begin(), recv_buffer.end());
      }
    }

    output.clear();
    output.reserve(rows + 1);
    output.push_back(static_cast<double>(rows));

    output.insert(output.end(), global_result.begin(), global_result.end());

    int output_size = static_cast<int>(output.size());
    for (int dst = 1; dst < size; ++dst) {
      MPI_Send(output.data(), output_size, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD);
    }

  } else {
    if (!local_result.empty()) {
      int local_size = static_cast<int>(local_result.size());
      MPI_Send(local_result.data(), local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    int result_size = 0;
    MPI_Status status;
    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &result_size);

    std::vector<double> recv_buffer(result_size);
    MPI_Recv(recv_buffer.data(), result_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    output = recv_buffer;
  }
}

bool StripedHorizontalMatrixVectorMPI::RunImpl() {
  const auto &input = GetInput();

  try {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = static_cast<int>(input[0]);
    int cols = static_cast<int>(input[1]);

    int rows_per_process = rows / size;
    int remainder = rows % size;

    int my_rows = rows_per_process;
    if (rank < remainder) {
      my_rows++;
    }

    int start_row = 0;
    for (int i = 0; i < rank; ++i) {
      int previous_rows = rows_per_process;
      if (i < remainder) {
        previous_rows++;
      }
      start_row += previous_rows;
    }

    std::vector<double> local_result = ProcessLocalRows(input, start_row, my_rows, cols, rows);

    GatherResults(rank, size, rows, rows_per_process, remainder, local_result, GetOutput());

    MPI_Barrier(MPI_COMM_WORLD);
    return true;

  } catch (...) {
    return false;
  }
}

bool StripedHorizontalMatrixVectorMPI::PostProcessingImpl() {
  const auto &output = GetOutput();
  if (output.empty()) {
    return false;
  }

  int result_size = static_cast<int>(output[0]);
  int rows = static_cast<int>(GetInput()[0]);

  return result_size == rows && output.size() == static_cast<size_t>(result_size) + 1;
}

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
