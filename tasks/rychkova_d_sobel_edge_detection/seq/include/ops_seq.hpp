#pragma once

#include <cstdint>
#include <vector>

#include "rychkova_d_sobel_edge_detection/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rychkova_d_sobel_edge_detection {

class SobelEdgeDetectionSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit SobelEdgeDetectionSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<uint8_t> gray_;
  std::vector<uint8_t> out_data_;
};

}  // namespace rychkova_d_sobel_edge_detection
