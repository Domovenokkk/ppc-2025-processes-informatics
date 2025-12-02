#pragma once

#include <cstdint>
#include <vector>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"
#include "task/include/task.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

class ContrastEnhancementMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit ContrastEnhancementMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

void FindGlobalMinMax(const std::vector<uint8_t> &local_pixels, unsigned char &global_min, unsigned char &global_max);

std::vector<uint8_t> ProcessLocalPixels(const std::vector<uint8_t> &local_pixels, uint8_t global_min,
                                        uint8_t global_max);

void GatherResults(int rank, int size, const std::vector<uint8_t> &local_result, int width, int height,
                   std::vector<uint8_t> &final_output);

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
