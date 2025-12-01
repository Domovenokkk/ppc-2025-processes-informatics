#include "mityaeva_d_striped_horizontal_matrix_vector/seq/include/ops_seq.hpp"

#include <algorithm>

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"

namespace mityaeva_d_striped_horizontal_matrix_vector {

StripedHorizontalMatrixVectorSEQ::StripedHorizontalMatrixVectorSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool StripedHorizontalMatrixVectorSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }
  if (GetInput().size() < 3) {
    return false;
  }

  int n = static_cast<int>(GetInput()[0]);
  int m = static_cast<int>(GetInput()[1]);

  if (n <= 0 || m <= 0) {
    return false;
  }
  if (GetInput().size() != static_cast<size_t>(2 + n * m + m)) {
    return false;
  }

  return true;
}

bool StripedHorizontalMatrixVectorSEQ::PreProcessingImpl() {
  n_ = static_cast<int>(GetInput()[0]);
  m_ = static_cast<int>(GetInput()[1]);

  matrix_linear_.resize(static_cast<size_t>(n_) * m_);

  size_t idx = 2;
  for (int i = 0; i < n_; ++i) {
    std::copy(GetInput().begin() + idx, GetInput().begin() + idx + m_,
              matrix_linear_.begin() + static_cast<size_t>(i) * m_);
    idx += m_;
  }

  vector_.resize(m_);
  for (int j = 0; j < m_; ++j) {
    vector_[j] = GetInput()[idx++];
  }

  GetOutput().assign(n_, 0.0);

  return true;
}

bool StripedHorizontalMatrixVectorSEQ::RunImpl() {
  const double *mat = matrix_linear_.data();
  const double *vec = vector_.data();
  double *out = GetOutput().data();

  for (int i = 0; i < n_; ++i) {
    const double *row = mat + static_cast<size_t>(i) * m_;
    double sum = 0.0;
    for (int j = 0; j < m_; ++j) {
      sum += row[j] * vec[j];
    }
    out[i] = sum;
  }

  return true;
}

bool StripedHorizontalMatrixVectorSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
