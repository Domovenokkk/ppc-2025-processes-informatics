#include "mityaeva_d_striped_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
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
  if (input.size() < 3) {
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

  size_t expected = 3 + static_cast<size_t>(rows) * static_cast<size_t>(cols) + static_cast<size_t>(cols);
  return input.size() == expected;
}

bool StripedHorizontalMatrixVectorMPI::PreProcessingImpl() {
  return true;
}

static void CalcRows(int rows, int size, int rank, int &my_rows, int &start_row) {
  int base = rows / size;
  int rem = rows % size;
  my_rows = base + (rank < rem);
  start_row = 0;
  for (int r = 0; r < rank; ++r) {
    start_row += base + (r < rem);
  }
}

static void BuildScatter(int rows, int cols, int size, std::vector<int> &counts, std::vector<int> &displs) {
  counts.resize(size);
  displs.resize(size);
  int base = rows / size;
  int rem = rows % size;
  int d = 0;
  for (int r = 0; r < size; ++r) {
    int r_rows = base + (r < rem);
    counts[r] = r_rows * cols;
    displs[r] = d;
    d += counts[r];
  }
}

static void BuildGather(int rows, int size, std::vector<int> &counts, std::vector<int> &displs) {
  counts.resize(size);
  displs.resize(size);
  int base = rows / size;
  int rem = rows % size;
  int d = 0;
  for (int r = 0; r < size; ++r) {
    int r_rows = base + (r < rem);
    counts[r] = r_rows;
    displs[r] = d;
    d += counts[r];
  }
}

bool StripedHorizontalMatrixVectorMPI::RunImpl() {
  const auto &input = GetInput();
  try {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = static_cast<int>(input[0]);
    int cols = static_cast<int>(input[1]);

    int my_rows = 0, start_row = 0;
    CalcRows(rows, size, rank, my_rows, start_row);

    std::vector<int> sendcounts, sdispls, recvcounts, rdispls;
    BuildScatter(rows, cols, size, sendcounts, sdispls);
    BuildGather(rows, size, recvcounts, rdispls);

    const size_t m_start = 3;
    const size_t v_start = m_start + static_cast<size_t>(rows) * static_cast<size_t>(cols);

    const double *A = input.data() + m_start;
    const double *x_all = input.data() + v_start;

    std::vector<double> local_A(static_cast<size_t>(my_rows) * cols);
    std::vector<double> x(cols);

    if (rank == 0) {
      std::copy(x_all, x_all + cols, x.begin());
    }

    MPI_Scatterv(rank == 0 ? A : nullptr, sendcounts.data(), sdispls.data(), MPI_DOUBLE, local_A.data(), my_rows * cols,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(x.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> y_local(my_rows, 0.0);
    for (int i = 0; i < my_rows; ++i) {
      double sum = 0.0;
      for (int j = 0; j < cols; ++j) {
        sum += local_A[static_cast<size_t>(i) * cols + j] * x[j];
      }
      y_local[i] = sum;
    }

    std::vector<double> y;
    if (rank == 0) {
      y.resize(rows);
    }

    MPI_Gatherv(y_local.data(), my_rows, MPI_DOUBLE, rank == 0 ? y.data() : nullptr, recvcounts.data(), rdispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      auto &out = GetOutput();
      out.clear();
      out.push_back(static_cast<double>(rows));
      out.insert(out.end(), y.begin(), y.end());
    }

    int out_size = 0;
    if (rank == 0) {
      out_size = static_cast<int>(GetOutput().size());
    }
    MPI_Bcast(&out_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
      GetOutput().assign(out_size, 0.0);
    }
    MPI_Bcast(GetOutput().data(), out_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
  int rows = static_cast<int>(GetInput()[0]);
  return static_cast<int>(output[0]) == rows && output.size() == static_cast<size_t>(rows) + 1;
}

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
