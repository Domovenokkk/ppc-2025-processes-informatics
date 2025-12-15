#pragma once

#include "rychkova_d_image_smoothing/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rychkova_d_image_smoothing {

class ImageSmoothingMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit ImageSmoothingMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace rychkova_d_image_smoothing
