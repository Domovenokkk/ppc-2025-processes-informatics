#pragma once

#include <vector>

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace mityaeva_d_striped_horizontal_matrix_vector {

class StripedHorizontalMatrixVectorMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit StripedHorizontalMatrixVectorMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int n_ = 0;
  int m_ = 0;
  int rank_ = 0;
  int world_size_ = 1;
  std::vector<std::vector<double>> local_rows_;
  std::vector<double> local_result_;
  std::vector<double> full_vector_;
};

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
