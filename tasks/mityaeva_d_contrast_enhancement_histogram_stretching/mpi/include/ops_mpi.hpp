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

  void CalculateDistribution(int rank, int size, int &my_pixels, int &my_offset);
  std::vector<uint8_t> GatherLocalPixels(const std::vector<uint8_t> &input, int my_pixels, int my_offset);
  std::pair<uint8_t, uint8_t> FindGlobalMinMax(const std::vector<uint8_t> &local_pixels);
  std::vector<uint8_t> ProcessLocalPixels(const std::vector<uint8_t> &local_pixels, uint8_t global_min,
                                          uint8_t global_max);
  void GatherResults(int rank, int size, const std::vector<uint8_t> &local_result, std::vector<uint8_t> &final_output);

  int width_{0};
  int height_{0};
  int total_pixels_{0};
};

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
