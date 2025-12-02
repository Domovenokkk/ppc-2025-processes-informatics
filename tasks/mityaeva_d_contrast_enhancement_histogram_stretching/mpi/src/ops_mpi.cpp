#include "mityaeva_d_contrast_enhancement_histogram_stretching/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"
#include "util/include/util.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

ContrastEnhancementMPI::ContrastEnhancementMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<uint8_t>{};
}

bool ContrastEnhancementMPI::ValidationImpl() {
  const auto &input = GetInput();

  if (input.size() < 3) {
    return false;
  }

  width_ = static_cast<int>(input[0]);
  height_ = static_cast<int>(input[1]);

  if (width_ <= 0 || height_ <= 0) {
    return false;
  }

  total_pixels_ = width_ * height_;

  return input.size() == static_cast<size_t>(total_pixels_) + 2;
}

bool ContrastEnhancementMPI::PreProcessingImpl() {
  return true;
}

void ContrastEnhancementMPI::CalculateDistribution(int rank, int size, int &my_pixels, int &my_offset) {
  int pixels_per_process = total_pixels_ / size;
  int remainder = total_pixels_ % size;

  my_pixels = pixels_per_process;
  if (rank < remainder) {
    my_pixels++;
  }

  my_offset = 0;
  for (int i = 0; i < rank; ++i) {
    int prev_pixels = pixels_per_process;
    if (i < remainder) {
      prev_pixels++;
    }
    my_offset += prev_pixels;
  }
}

std::vector<uint8_t> ContrastEnhancementMPI::GatherLocalPixels(const std::vector<uint8_t> &input, int my_pixels,
                                                               int my_offset) {
  std::vector<uint8_t> local_pixels;

  if (my_pixels > 0) {
    local_pixels.reserve(my_pixels);
    size_t start_idx = 2 + my_offset;

    for (int i = 0; i < my_pixels; ++i) {
      if (start_idx + i < input.size()) {
        local_pixels.push_back(input[start_idx + i]);
      }
    }
  }

  return local_pixels;
}

std::pair<uint8_t, uint8_t> ContrastEnhancementMPI::FindGlobalMinMax(const std::vector<uint8_t> &local_pixels) {
  uint8_t local_min = 255;
  uint8_t local_max = 0;

  for (uint8_t pixel : local_pixels) {
    if (pixel < local_min) {
      local_min = pixel;
    }
    if (pixel > local_max) {
      local_max = pixel;
    }
  }

  uint8_t global_min, global_max;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);

  return std::make_pair(global_min, global_max);
}

std::vector<uint8_t> ContrastEnhancementMPI::ProcessLocalPixels(const std::vector<uint8_t> &local_pixels,
                                                                uint8_t global_min, uint8_t global_max) {
  std::vector<uint8_t> local_result;

  if (global_min == global_max) {
    local_result = local_pixels;
    return local_result;
  }

  double scale = 255.0 / static_cast<double>(global_max - global_min);
  local_result.reserve(local_pixels.size());

  for (uint8_t pixel : local_pixels) {
    double stretched_value = static_cast<double>(pixel - global_min) * scale;
    int rounded_value = static_cast<int>(std::round(stretched_value));

    if (rounded_value < 0) {
      rounded_value = 0;
    }
    if (rounded_value > 255) {
      rounded_value = 255;
    }

    local_result.push_back(static_cast<uint8_t>(rounded_value));
  }

  return local_result;
}

void ContrastEnhancementMPI::GatherResults(int rank, int size, const std::vector<uint8_t> &local_result,
                                           std::vector<uint8_t> &final_output) {
  std::vector<int> recv_counts(size, 0);
  std::vector<int> displs(size, 0);

  int local_size = static_cast<int>(local_result.size());
  MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    int total_size = 0;
    for (int i = 0; i < size; ++i) {
      displs[i] = total_size;
      total_size += recv_counts[i];
    }

    if (total_size != total_pixels_) {
      return;
    }

    final_output.reserve(total_size + 2);
    final_output.push_back(static_cast<uint8_t>(width_));
    final_output.push_back(static_cast<uint8_t>(height_));
    final_output.resize(total_size + 2);
  }

  MPI_Gatherv(local_result.data(), local_size, MPI_UNSIGNED_CHAR, (rank == 0) ? final_output.data() + 2 : nullptr,
              recv_counts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  int final_output_size = 0;
  if (rank == 0) {
    final_output_size = static_cast<int>(final_output.size());
  }

  MPI_Bcast(&final_output_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    final_output.resize(final_output_size);
  }

  MPI_Bcast(final_output.data(), final_output_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
}

bool ContrastEnhancementMPI::RunImpl() {
  try {
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto &input = GetInput();

    int my_pixels;
    int my_offset;
    CalculateDistribution(rank, size, my_pixels, my_offset);

    std::vector<uint8_t> local_pixels = GatherLocalPixels(input, my_pixels, my_offset);

    auto min_max_pair = FindGlobalMinMax(local_pixels);
    uint8_t global_min = min_max_pair.first;
    uint8_t global_max = min_max_pair.second;
    std::vector<uint8_t> local_result = ProcessLocalPixels(local_pixels, global_min, global_max);

    std::vector<uint8_t> final_output;
    GatherResults(rank, size, local_result, final_output);
    GetOutput() = std::move(final_output);

    MPI_Barrier(MPI_COMM_WORLD);
    return true;

  } catch (...) {
    return false;
  }
}

bool ContrastEnhancementMPI::PostProcessingImpl() {
  const auto &output = GetOutput();

  if (output.empty()) {
    return false;
  }

  if (output.size() < 2) {
    return false;
  }

  int out_width = static_cast<int>(output[0]);
  int out_height = static_cast<int>(output[1]);

  if (out_width != width_ || out_height != height_) {
    return false;
  }

  if (output.size() != static_cast<size_t>(total_pixels_) + 2) {
    return false;
  }

  if (output.size() <= 2) {
    return false;
  }

  return true;
}

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
