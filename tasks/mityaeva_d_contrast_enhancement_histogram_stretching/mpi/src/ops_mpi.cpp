#include "mityaeva_d_contrast_enhancement_histogram_stretching/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"

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

  int width = static_cast<int>(input[0]);
  int height = static_cast<int>(input[1]);

  if (width <= 0 || height <= 0) {
    return false;
  }

  int total_pixels = width * height;

  return input.size() == static_cast<size_t>(total_pixels) + 2;
}

bool ContrastEnhancementMPI::PreProcessingImpl() {
  return true;
}

void FindGlobalMinMax(const std::vector<uint8_t> &local_pixels, uint8_t &local_min, uint8_t &local_max,
                      uint8_t &global_min, uint8_t &global_max) {
  local_min = kMaxPixelValue;
  local_max = kMinPixelValue;

  for (uint8_t pixel : local_pixels) {
    local_min = std::min(pixel, local_min);
    local_max = std::max(pixel, local_max);
  }

  MPI_Allreduce(&local_min, &global_min, 1, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);
}

std::vector<uint8_t> ProcessLocalPixels(const std::vector<uint8_t> &local_pixels, uint8_t global_min,
                                        uint8_t global_max) {
  std::vector<uint8_t> local_result;
  local_result.reserve(local_pixels.size());

  if (global_min == global_max) {
    local_result = local_pixels;
    return local_result;
  }

  double scale = static_cast<double>(kMaxPixelValue - kMinPixelValue) / static_cast<double>(global_max - global_min);

  for (uint8_t pixel : local_pixels) {
    double new_value = static_cast<double>(pixel - global_min) * scale;

    double clamped_value = new_value + 0.5;
    if (clamped_value < 0.0) {
      clamped_value = 0.0;
    }
    if (clamped_value > 255.0) {
      clamped_value = 255.0;
    }

    uint8_t enhanced_pixel = static_cast<uint8_t>(clamped_value);
    local_result.push_back(enhanced_pixel);
  }

  return local_result;
}

void GatherResults(int rank, int size, const std::vector<uint8_t> &local_result, int width, int height,
                   std::vector<uint8_t> &final_output) {
  int local_size = static_cast<int>(local_result.size());
  std::vector<int> recv_counts(size, 0);
  std::vector<int> displs(size, 0);

  MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }

    int total_size = displs[size - 1] + recv_counts[size - 1];
    if (total_size != width * height) {
      return;
    }

    final_output.clear();
    final_output.reserve(total_size + 2);
    final_output.push_back(static_cast<uint8_t>(width));
    final_output.push_back(static_cast<uint8_t>(height));

    final_output.resize(total_size + 2);
  }

  MPI_Gatherv(local_result.data(), local_size, MPI_UNSIGNED_CHAR, (rank == 0) ? final_output.data() + 2 : nullptr,
              recv_counts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  int total_output_size = 0;
  if (rank == 0) {
    total_output_size = static_cast<int>(final_output.size());
  }

  MPI_Bcast(&total_output_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    final_output.resize(total_output_size);
  }

  MPI_Bcast(final_output.data(), total_output_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
}

bool ContrastEnhancementMPI::RunImpl() {
  const auto &input = GetInput();

  try {
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = static_cast<int>(input[0]);
    int height = static_cast<int>(input[1]);
    int total_pixels = width * height;

    int pixels_per_process = total_pixels / size;
    int remainder = total_pixels % size;

    int my_pixels = pixels_per_process;
    if (rank < remainder) {
      my_pixels++;
    }

    int my_offset = 0;
    for (int i = 0; i < rank; ++i) {
      int prev_pixels = pixels_per_process;
      if (i < remainder) {
        prev_pixels++;
      }
      my_offset += prev_pixels;
    }

    if (my_pixels == 0) {
      my_offset = 0;
    }

    std::vector<uint8_t> local_pixels;
    if (my_pixels > 0 && my_offset + my_pixels <= total_pixels) {
      local_pixels.reserve(my_pixels);
      size_t start_idx = 2 + my_offset;

      for (int i = 0; i < my_pixels; ++i) {
        local_pixels.push_back(input[start_idx + i]);
      }
    }

    uint8_t local_min;
    uint8_t local_max;
    uint8_t global_min;
    uint8_t global_max;
    FindGlobalMinMax(local_pixels, local_min, local_max, global_min, global_max);

    std::vector<uint8_t> local_result = ProcessLocalPixels(local_pixels, global_min, global_max);

    std::vector<uint8_t> final_output;
    GatherResults(rank, size, local_result, width, height, final_output);

    GetOutput() = final_output;

    MPI_Barrier(MPI_COMM_WORLD);
    return true;

  } catch (...) {
    return false;
  }
}

bool ContrastEnhancementMPI::PostProcessingImpl() {
  const auto &output = GetOutput();

  if (output.size() < 2) {
    return false;
  }

  int out_width = static_cast<int>(output[0]);
  int out_height = static_cast<int>(output[1]);

  int in_width = static_cast<int>(GetInput()[0]);
  int in_height = static_cast<int>(GetInput()[1]);

  if (out_width != in_width || out_height != in_height) {
    return false;
  }

  int total_pixels = out_width * out_height;
  if (output.size() != static_cast<size_t>(total_pixels) + 2) {
    return false;
  }

  for (size_t i = 2; i < output.size(); ++i) {
    uint8_t pixel = output[i];
    if (pixel < kMinPixelValue || pixel > kMaxPixelValue) {
      return false;
    }
  }

  return true;
}

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
