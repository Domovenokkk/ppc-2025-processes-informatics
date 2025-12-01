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
  // Получаем размеры матрицы
  if (rank_ == 0) {
    n_ = static_cast<int>(GetInput()[0]);
    m_ = static_cast<int>(GetInput()[1]);
  }

  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Рассылаем вектор b всем процессам
  full_vector_.resize(m_);
  if (rank_ == 0) {
    size_t vector_start_index = 2 + n_ * m_;
    for (int j = 0; j < m_; ++j) {
      full_vector_[j] = GetInput()[vector_start_index + j];
    }
  }
  MPI_Bcast(full_vector_.data(), m_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Распределяем строки матрицы по процессам
  int rows_per_process = n_ / world_size_;
  int remainder = n_ % world_size_;

  // Определяем количество строк для текущего процесса
  int local_row_count = rows_per_process;
  if (rank_ < remainder) {
    local_row_count++;
  }

  // Выделяем память для локальных строк
  local_rows_.resize(local_row_count, std::vector<double>(m_));
  local_result_.resize(local_row_count, 0.0);

  // Процесс 0 распределяет данные
  if (rank_ == 0) {
    // Определяем смещения для каждого процесса
    std::vector<int> process_offsets(world_size_);
    std::vector<int> process_counts(world_size_);

    int offset = 0;
    for (int proc = 0; proc < world_size_; ++proc) {
      int proc_rows = rows_per_process;
      if (proc < remainder) {
        proc_rows++;
      }
      process_counts[proc] = proc_rows;
      process_offsets[proc] = offset;
      offset += proc_rows;
    }

    // Процесс 0 копирует свои строки
    for (int i = 0; i < local_row_count; ++i) {
      int global_row = process_offsets[0] + i;
      for (int j = 0; j < m_; ++j) {
        local_rows_[i][j] = GetInput()[2 + global_row * m_ + j];
      }
    }

    // Отправляем строки другим процессам
    for (int dest_rank = 1; dest_rank < world_size_; ++dest_rank) {
      int dest_row_count = process_counts[dest_rank];
      if (dest_row_count > 0) {
        std::vector<double> dest_rows_data(dest_row_count * m_);
        for (int i = 0; i < dest_row_count; ++i) {
          int global_row = process_offsets[dest_rank] + i;
          for (int j = 0; j < m_; ++j) {
            dest_rows_data[i * m_ + j] = GetInput()[2 + global_row * m_ + j];
          }
        }
        MPI_Send(dest_rows_data.data(), dest_row_count * m_, MPI_DOUBLE, dest_rank, 0, MPI_COMM_WORLD);
      }
    }
  } else if (local_row_count > 0) {
    // Другие процессы получают свои строки
    std::vector<double> received_data(local_row_count * m_);
    MPI_Recv(received_data.data(), local_row_count * m_, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Распаковываем данные в локальные строки
    for (int i = 0; i < local_row_count; ++i) {
      for (int j = 0; j < m_; ++j) {
        local_rows_[i][j] = received_data[i * m_ + j];
      }
    }
  }

  // Инициализируем выходной вектор ТОЛЬКО на процессе 0
  if (rank_ == 0) {
    GetOutput().resize(n_, 0.0);
  } else {
    GetOutput().clear();
  }

  return true;
}

bool StripedHorizontalMatrixVectorMPI::RunImpl() {
  // Каждый процесс вычисляет свою часть результата
  for (size_t i = 0; i < local_rows_.size(); ++i) {
    double sum = 0.0;
    for (int j = 0; j < m_; ++j) {
      sum += local_rows_[i][j] * full_vector_[j];
    }
    local_result_[i] = sum;
  }

  // Собираем результаты на процессе 0
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

  // Собираем результаты только если есть что собирать
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

  // Для процессов кроме 0 проверяем, что выход пустой
  if (!GetOutput().empty()) {
    GetOutput().clear();
  }

  return true;
}

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
