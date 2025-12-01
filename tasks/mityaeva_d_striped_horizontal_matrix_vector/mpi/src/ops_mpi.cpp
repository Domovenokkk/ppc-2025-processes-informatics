#include "mityaeva_d_striped_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

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
    if (GetInput().empty() || GetInput().size() < 3) {
      is_valid = false;
    } else {
      int n = static_cast<int>(GetInput()[0]);
      int m = static_cast<int>(GetInput()[1]);
      if (n <= 0 || m <= 0 || GetInput().size() != static_cast<size_t>(2 + n * m + m)) {
        is_valid = false;
      }
      // Проверка, что количество процессов не превышает количество строк
      if (world_size_ > n) {
        is_valid = false;
      }
    }
  }

  int is_valid_int = is_valid ? 1 : 0;
  MPI_Bcast(&is_valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return (is_valid_int != 0);
}

bool StripedHorizontalMatrixVectorMPI::PreProcessingImpl() {
  // Размеры матрицы
  if (rank_ == 0) {
    n_ = static_cast<int>(GetInput()[0]);
    m_ = static_cast<int>(GetInput()[1]);
  }
  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Проверка корректности размеров
  if (n_ <= 0 || m_ <= 0) {
    return false;
  }

  // Вектор
  full_vector_.assign(m_, 0.0);
  if (rank_ == 0) {
    size_t vec_start = 2 + static_cast<size_t>(n_) * m_;
    for (int j = 0; j < m_; ++j) {
      full_vector_[j] = GetInput()[vec_start + j];
    }
  }
  if (m_ > 0) {
    MPI_Bcast(full_vector_.data(), m_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // Распределение строк
  std::vector<int> rows_per_proc(world_size_, 0);
  int base = n_ / world_size_;
  int rem = n_ % world_size_;
  for (int i = 0; i < world_size_; ++i) {
    rows_per_proc[i] = base + (i < rem ? 1 : 0);
  }

  int local_rows_count = rows_per_proc[rank_];
  if (local_rows_count > 0) {
    local_rows_.assign(local_rows_count, std::vector<double>(m_));
    local_result_.assign(local_rows_count, 0.0);
  } else {
    // Если процессу не досталось строк, создаем пустые векторы
    local_rows_.clear();
    local_result_.clear();
  }

  // Подготовка sendcounts и displs для Scatterv
  std::vector<int> sendcounts(world_size_, 0);
  std::vector<int> displs(world_size_, 0);
  int offset = 0;
  for (int i = 0; i < world_size_; ++i) {
    sendcounts[i] = rows_per_proc[i] * m_;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  // Подготовка буфера для приема данных
  std::vector<double> recvbuf;
  if (local_rows_count > 0) {
    recvbuf.assign(local_rows_count * m_, 0.0);
  }

  // Подготовка плоского буфера для Scatterv на root процессе
  std::vector<double> flat_mat;
  if (rank_ == 0) {
    flat_mat.reserve(static_cast<size_t>(n_) * m_);
    for (int r = 0; r < n_; ++r) {
      for (int c = 0; c < m_; ++c) {
        flat_mat.push_back(GetInput()[2 + r * m_ + c]);
      }
    }
  }

  // Распределение строк матрицы по процессам
  MPI_Scatterv(rank_ == 0 ? flat_mat.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
               local_rows_count > 0 ? recvbuf.data() : nullptr,
               local_rows_count > 0 ? static_cast<int>(recvbuf.size()) : 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Распаковка recvbuf в local_rows_
  if (local_rows_count > 0) {
    for (int i = 0; i < local_rows_count; ++i) {
      for (int j = 0; j < m_; ++j) {
        local_rows_[i][j] = recvbuf[i * m_ + j];
      }
    }
  }

  // Выходной вектор на root
  if (rank_ == 0) {
    GetOutput().assign(n_, 0.0);
  }

  return true;
}

bool StripedHorizontalMatrixVectorMPI::RunImpl() {
  // Локальное умножение строк на вектор
  for (size_t i = 0; i < local_rows_.size(); ++i) {
    double sum = 0.0;
    for (int j = 0; j < m_; ++j) {
      sum += local_rows_[i][j] * full_vector_[j];
    }
    local_result_[i] = sum;
  }

  // Gatherv для сбора результатов
  std::vector<int> rows_per_proc(world_size_, 0);
  int base = n_ / world_size_;
  int rem = n_ % world_size_;
  for (int i = 0; i < world_size_; ++i) {
    rows_per_proc[i] = base + (i < rem ? 1 : 0);
  }

  std::vector<int> recvcounts(world_size_, 0);
  std::vector<int> displs(world_size_, 0);
  int offset = 0;
  for (int i = 0; i < world_size_; ++i) {
    recvcounts[i] = rows_per_proc[i];
    displs[i] = offset;
    offset += recvcounts[i];
  }

  // Сбор результатов на root процессе
  MPI_Gatherv(local_result_.empty() ? nullptr : local_result_.data(), static_cast<int>(local_result_.size()),
              MPI_DOUBLE, rank_ == 0 ? GetOutput().data() : nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  // УБРАН MPI_Barrier - он не нужен и может вызывать проблемы

  return true;
}

bool StripedHorizontalMatrixVectorMPI::PostProcessingImpl() {
  // Проверяем результат только на root процессе
  if (rank_ == 0) {
    // Убеждаемся, что выходной вектор имеет правильный размер
    if (GetOutput().size() != static_cast<size_t>(n_)) {
      GetOutput().assign(n_, 0.0);
      return false;
    }
    return !GetOutput().empty();
  }

  // На остальных процессах просто возвращаем true
  // Не очищаем GetOutput(), так как на этих процессах он и так пуст
  return true;
}

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
