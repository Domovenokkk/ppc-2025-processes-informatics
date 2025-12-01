#include "mityaeva_d_striped_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdlib>
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

  // Передаём флаг валидности как int — более переносимо
  int is_valid_int = is_valid ? 1 : 0;
  MPI_Bcast(&is_valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  is_valid = (is_valid_int != 0);

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
  full_vector_.assign(m_, 0.0);
  if (rank_ == 0) {
    size_t vector_start_index = 2 + static_cast<size_t>(n_) * static_cast<size_t>(m_);
    for (int j = 0; j < m_; ++j) {
      full_vector_[j] = GetInput()[vector_start_index + j];
    }
  }
  if (m_ > 0) {
    MPI_Bcast(full_vector_.data(), m_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // Распределяем строки матрицы по процессам — подготовим rows_per_proc
  std::vector<int> rows_per_proc(world_size_, 0);
  int base = 0;
  int rem = 0;
  if (world_size_ > 0) {
    base = (n_ > 0) ? (n_ / world_size_) : 0;
    rem = (n_ > 0) ? (n_ % world_size_) : 0;
  }
  for (int i = 0; i < world_size_; ++i) {
    rows_per_proc[i] = base + (i < rem ? 1 : 0);
  }
  int local_row_count = rows_per_proc[rank_];

  // Подготовим локальное хранилище
  local_rows_.assign(local_row_count, std::vector<double>(m_));
  local_result_.assign(local_row_count, 0.0);

  // Подготовим параметры для MPI_Scatterv: sendcounts и displs в элементах типа double
  std::vector<int> sendcounts(world_size_, 0);
  std::vector<int> displs(world_size_, 0);
  for (int i = 0, offset = 0; i < world_size_; ++i) {
    // количество элементов (double) для процесса i = rows_per_proc[i] * m_
    long long elems = static_cast<long long>(rows_per_proc[i]) * static_cast<long long>(m_);
    // безопасно привести к int (MPI требует int), предполагается что числа небольшие в задачах теста
    sendcounts[i] = static_cast<int>(elems);
    displs[i] = offset;
    offset += sendcounts[i];
  }

  // На всех процессах создаём recvbuf плоской формы, даже на root для простоты
  std::vector<double> recvbuf;
  if (local_row_count > 0 && m_ > 0) {
    recvbuf.resize(static_cast<size_t>(local_row_count) * static_cast<size_t>(m_));
  }

  if (rank_ == 0) {
    // Формируем плоский буфер матрицы row-major из GetInput()
    std::vector<double> flat_mat;
    if (n_ > 0 && m_ > 0) {
      flat_mat.reserve(static_cast<size_t>(n_) * static_cast<size_t>(m_));
      for (int r = 0; r < n_; ++r) {
        for (int c = 0; c < m_; ++c) {
          flat_mat.push_back(GetInput()[2 + r * m_ + c]);
        }
      }
    }

    // Вызовем MPI_Scatterv: root отправляет flat_mat, все получают в recvbuf
    // Учтём случаи, когда sendcounts[i] == 0
    if (n_ > 0 && m_ > 0) {
      MPI_Scatterv(flat_mat.empty() ? nullptr : flat_mat.data(),  // sendbuf
                   sendcounts.data(),                             // sendcounts (int)
                   displs.data(),                                 // displacements (int)
                   MPI_DOUBLE,
                   (recvbuf.empty() ? nullptr : recvbuf.data()),  // recvbuf
                   static_cast<int>(recvbuf.size()),              // recvcount (int)
                   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
      // если матрицы нет (n_==0 || m_==0), всё равно вызываем Scatterv с нулями
      MPI_Scatterv(nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  } else {
    // non-root: получаем часть матрицы
    if (local_row_count > 0 && m_ > 0) {
      MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, recvbuf.data(), static_cast<int>(recvbuf.size()), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    } else {
      // ничего не принимаем
      MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  }

  // Распакуем recvbuf в local_rows_
  if (!recvbuf.empty()) {
    for (int i = 0; i < local_row_count; ++i) {
      for (int j = 0; j < m_; ++j) {
        local_rows_[i][j] = recvbuf[i * m_ + j];
      }
    }
  }

  // Инициализируем выходной вектор ТОЛЬКО на процессе 0
  if (rank_ == 0) {
    GetOutput().assign(std::max(0, n_), 0.0);
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

  // Подготовим recv_counts и displacements для Gatherv (в элементах типа double)
  std::vector<int> recv_counts(world_size_, 0);
  std::vector<int> displacements(world_size_, 0);
  for (int i = 0, disp = 0; i < world_size_; ++i) {
    int rows = (n_ > 0) ? (n_ / world_size_ + (i < (n_ % world_size_) ? 1 : 0)) : 0;
    recv_counts[i] = rows;  // каждый процесс возвращает rows элементов (по 1 double на строку)
    displacements[i] = disp;
    disp += recv_counts[i];
  }

  int sendcount = static_cast<int>(local_result_.size());  // число double-ов, которые шлём
  if (n_ > 0) {
    MPI_Gatherv((sendcount > 0 ? local_result_.data() : nullptr), sendcount, MPI_DOUBLE,
                (rank_ == 0 ? GetOutput().data() : nullptr), recv_counts.data(), displacements.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
  } else {
    // ничего не собираем, но вызовем Gatherv с нулевыми значениями (безопасно)
    MPI_Gatherv(nullptr, 0, MPI_DOUBLE, nullptr, recv_counts.data(), displacements.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool StripedHorizontalMatrixVectorMPI::PostProcessingImpl() {
  if (rank_ == 0) {
    return !GetOutput().empty() && static_cast<int>(GetOutput().size()) == n_;
  }

  // Для процессов кроме 0 проверяем, что выход пустой
  if (!GetOutput().empty()) {
    GetOutput().clear();
  }

  return true;
}

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
