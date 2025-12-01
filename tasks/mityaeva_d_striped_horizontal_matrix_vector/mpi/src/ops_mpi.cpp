#include "mityaeva_d_striped_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"
#include "util/include/util.hpp"

namespace mityaeva_d_striped_horizontal_matrix_vector {

StripedHorizontalMatrixVectorMPI::StripedHorizontalMatrixVectorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  n_ = 0;
  m_ = 0;
  rank_ = 0;
  world_size_ = 1;
}

bool StripedHorizontalMatrixVectorMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  bool is_valid = true;

  if (rank_ == 0) {
    if (GetInput().empty()) {
      is_valid = false;
    }

    if (GetInput().size() < 3) {
      is_valid = false;
    }

    if (is_valid) {
      int n = static_cast<int>(GetInput()[0]);
      int m = static_cast<int>(GetInput()[1]);

      if (n <= 0 || m <= 0) {
        is_valid = false;
      }

      if (GetInput().size() != static_cast<size_t>(2 + n * m + m)) {
        is_valid = false;
      }
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

  return is_valid;
}

bool StripedHorizontalMatrixVectorMPI::PreProcessingImpl() {
  if (rank_ == 0) {
    n_ = static_cast<int>(GetInput()[0]);
    m_ = static_cast<int>(GetInput()[1]);
  }

  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  full_vector_.resize(m_);
  if (rank_ == 0) {
    size_t vector_start_index = 2 + n_ * m_;
    for (int j = 0; j < m_; ++j) {
      full_vector_[j] = GetInput()[vector_start_index + j];
    }
  }
  MPI_Bcast(full_vector_.data(), m_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int rows_per_process = n_ / world_size_;
  int remainder = n_ % world_size_;

  int local_row_count = rows_per_process;
  if (rank_ < remainder) {
    local_row_count++;
  }

  int start_row = 0;
  for (int i = 0; i < rank_; ++i) {
    int proc_rows = rows_per_process;
    if (i < remainder) {
      proc_rows++;
    }
    start_row += proc_rows;
  }

  local_rows_.resize(local_row_count, std::vector<double>(m_));
  local_result_.resize(local_row_count, 0.0);

  if (rank_ == 0) {
    for (int i = 0; i < local_row_count; ++i) {
      for (int j = 0; j < m_; ++j) {
        local_rows_[i][j] = GetInput()[2 + (start_row + i) * m_ + j];
      }
    }

    for (int dest_rank = 1; dest_rank < world_size_; ++dest_rank) {
      int dest_row_count = rows_per_process;
      if (dest_rank < remainder) {
        dest_row_count++;
      }
      int dest_start_row = 0;
      for (int i = 0; i < dest_rank; ++i) {
        int proc_rows = rows_per_process;
        if (i < remainder) {
          proc_rows++;
        }
        dest_start_row += proc_rows;
      }
      std::vector<double> dest_rows_data(dest_row_count * m_);
      for (int i = 0; i < dest_row_count; ++i) {
        for (int j = 0; j < m_; ++j) {
          dest_rows_data[i * m_ + j] = GetInput()[2 + (dest_start_row + i) * m_ + j];
        }
      }
      MPI_Send(dest_rows_data.data(), dest_row_count * m_, MPI_DOUBLE, dest_rank, 0, MPI_COMM_WORLD);
    }
  } else {
    std::vector<double> received_data(local_row_count * m_);
    MPI_Recv(received_data.data(), local_row_count * m_, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < local_row_count; ++i) {
      for (int j = 0; j < m_; ++j) {
        local_rows_[i][j] = received_data[i * m_ + j];
      }
    }
  }
  if (rank_ == 0) {
    GetOutput().resize(n_, 0.0);
  } else {
    GetOutput().clear();
  }

  return true;
}

bool StripedHorizontalMatrixVectorMPI::RunImpl() {
  for (int i = 0; i < local_rows_.size(); ++i) {
    double sum = 0.0;
    for (int j = 0; j < m_; ++j) {
      sum += local_rows_[i][j] * full_vector_[j];
    }
    local_result_[i] = sum;
  }

  std::vector<int> recv_counts(world_size_);
  std::vector<int> displacements(world_size_);

  int rows_per_process = n_ / world_size_;
  int remainder = n_ % world_size_;

  int displacement = 0;
  for (int i = 0; i < world_size_; ++i) {
    recv_counts[i] = rows_per_process;
    if (i < remainder) {
      recv_counts[i]++;
    }
    displacements[i] = displacement;
    displacement += recv_counts[i];
  }

  if (n_ > 0) {
    MPI_Gatherv(local_result_.data(), local_result_.size(), MPI_DOUBLE, GetOutput().data(), recv_counts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool StripedHorizontalMatrixVectorMPI::PostProcessingImpl() {
  if (rank_ == 0) {
    return !GetOutput().empty() && GetOutput().size() == static_cast<size_t>(n_);
  }

  if (!GetOutput().empty()) {
    GetOutput().clear();
  }

  return true;
}

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
