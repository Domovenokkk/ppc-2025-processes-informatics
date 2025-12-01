#include <vector>

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"
#include "mityaeva_d_striped_horizontal_matrix_vector/seq/include/ops_seq.hpp"
#include "util/include/util.hpp"

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

  matrix_.resize(n_, std::vector<double>(m_));

  size_t index = 2;
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < m_; ++j) {
      matrix_[i][j] = GetInput()[index++];
    }
  }

  vector_.resize(m_);
  for (int j = 0; j < m_; ++j) {
    vector_[j] = GetInput()[index++];
  }

  GetOutput().resize(n_, 0.0);
  return true;
}

bool StripedHorizontalMatrixVectorSEQ::RunImpl() {
  for (int i = 0; i < n_; ++i) {
    double sum = 0.0;
    for (int j = 0; j < m_; ++j) {
      sum += matrix_[i][j] * vector_[j];
    }
    GetOutput()[i] = sum;
  }

  return true;
}

bool StripedHorizontalMatrixVectorSEQ::PostProcessingImpl() {
  return !GetOutput().empty() && GetOutput().size() == static_cast<size_t>(n_);
}

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
