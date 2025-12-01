#include "mityaeva_d_striped_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <limits>
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

  int is_valid_int = is_valid ? 1 : 0;
  MPI_Bcast(&is_valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  is_valid = (is_valid_int != 0);

  return is_valid;
}

bool StripedHorizontalMatrixVectorMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  if (rank_ == 0) {
    n_ = static_cast<int>(GetInput()[0]);
    m_ = static_cast<int>(GetInput()[1]);
  }

  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  full_vector_.assign(m_, 0.0);
  if (rank_ == 0 && m_ > 0) {
    size_t vector_start_index = 2 + static_cast<size_t>(n_) * static_cast<size_t>(m_);
    for (int j = 0; j < m_; ++j) {
      full_vector_[j] = GetInput()[vector_start_index + j];
    }
  }
  if (m_ > 0) {
    MPI_Bcast(full_vector_.data(), m_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  std::vector<int> rows_per_proc(world_size_, 0);
  int base = 0;
  int rem = 0;
  if (world_size_ > 0 && n_ > 0) {
    base = n_ / world_size_;
    rem = n_ % world_size_;
  }
  for (int i = 0; i < world_size_; ++i) {
    rows_per_proc[i] = base + (i < rem ? 1 : 0);
  }
  int local_row_count = rows_per_proc[rank_];

  local_rows_.assign(local_row_count, std::vector<double>(m_));
  local_result_.assign(local_row_count, 0.0);

  std::vector<int> sendcounts(world_size_, 0);
  std::vector<int> displs(world_size_, 0);
  {
    int offset = 0;
    for (int i = 0; i < world_size_; ++i) {
      long long elems = static_cast<long long>(rows_per_proc[i]) * static_cast<long long>(m_);
      sendcounts[i] = static_cast<int>(elems);
      displs[i] = offset;
      offset += sendcounts[i];
    }
  }

  std::vector<double> recvbuf;
  if (local_row_count > 0 && m_ > 0) {
    recvbuf.resize(static_cast<size_t>(local_row_count) * static_cast<size_t>(m_));
  }

  std::vector<double> flat_mat;
  if (rank_ == 0 && n_ > 0 && m_ > 0) {
    flat_mat.reserve(static_cast<size_t>(n_) * static_cast<size_t>(m_));
    for (int r = 0; r < n_; ++r) {
      for (int c = 0; c < m_; ++c) {
        flat_mat.push_back(GetInput()[2 + r * m_ + c]);
      }
    }
  }

  int recvcount = static_cast<int>(recvbuf.size());
  MPI_Scatterv(rank_ == 0 ? (flat_mat.empty() ? nullptr : flat_mat.data()) : nullptr, sendcounts.data(), displs.data(),
               MPI_DOUBLE, (recvbuf.empty() ? nullptr : recvbuf.data()), recvcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (!recvbuf.empty()) {
    for (int i = 0; i < local_row_count; ++i) {
      for (int j = 0; j < m_; ++j) {
        local_rows_[i][j] = recvbuf[static_cast<size_t>(i) * m_ + j];
      }
    }
  }

  if (rank_ == 0) {
    GetOutput().assign(std::max(0, n_), 0.0);
  } else {
    GetOutput().clear();
  }

  return true;
}

bool StripedHorizontalMatrixVectorMPI::RunImpl() {
  for (size_t i = 0; i < local_rows_.size(); ++i) {
    double sum = 0.0;
    for (int j = 0; j < m_; ++j) {
      sum += local_rows_[i][j] * full_vector_[j];
    }
    local_result_[i] = sum;
  }

  std::vector<int> recv_counts(world_size_, 0);
  std::vector<int> displacements(world_size_, 0);
  {
    int disp = 0;
    for (int i = 0; i < world_size_; ++i) {
      int rows = (n_ > 0) ? (n_ / world_size_ + (i < (n_ % world_size_) ? 1 : 0)) : 0;
      recv_counts[i] = rows;
      displacements[i] = disp;
      disp += recv_counts[i];
    }
  }

  int sendcount = static_cast<int>(local_result_.size());
  MPI_Gatherv((sendcount > 0 ? local_result_.data() : nullptr), sendcount, MPI_DOUBLE,
              (rank_ == 0 ? (GetOutput().empty() ? nullptr : GetOutput().data()) : nullptr), recv_counts.data(),
              displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}
// da
bool StripedHorizontalMatrixVectorMPI::PostProcessingImpl() {
  if (rank_ == 0) {
    return !GetOutput().empty() && static_cast<int>(GetOutput().size()) == n_;
  }

  if (!GetOutput().empty()) {
    GetOutput().clear();
  }

  return true;
}

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
