#include "mityaeva_d_contrast_enhancement_histogram_stretching/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

static constexpr int kComputeRepeats = 200;

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
  return input.size() == static_cast<size_t>(width * height) + 2;
}

bool ContrastEnhancementMPI::PreProcessingImpl() {
  return true;
}

void FindGlobalMinMax(const std::vector<uint8_t> &local_pixels, unsigned char &global_min, unsigned char &global_max) {
  auto local_min = kMaxPixelValue;
  auto local_max = kMinPixelValue;

  for (auto pixel : local_pixels) {
    auto value = static_cast<unsigned char>(pixel);
    local_min = std::min(local_min, value);
    local_max = std::max(local_max, value);
  }

  MPI_Allreduce(&local_min, &global_min, 1, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);
}

std::vector<uint8_t> ProcessLocalPixels(const std::vector<uint8_t> &local_pixels, unsigned char global_min,
                                        unsigned char global_max) {
  std::vector<uint8_t> result;
  result.reserve(local_pixels.size());

  if (global_min == global_max) {
    volatile double sink = 0.0;
    for (int repeat_index = 0; repeat_index < kComputeRepeats; ++repeat_index) {
      for (auto pixel : local_pixels) {
        sink += pixel;
      }
    }
    result = local_pixels;
    return result;
  }

  double scale = static_cast<double>(kMaxPixelValue - kMinPixelValue) / static_cast<double>(global_max - global_min);

  std::vector<uint8_t> temp;
  temp.reserve(local_pixels.size());

  for (auto pixel : local_pixels) {
    double value = static_cast<double>(pixel - global_min) * scale;
    value = std::min(std::max(value + 0.5, 0.0), 255.0);
    temp.push_back(static_cast<uint8_t>(value));
  }

  volatile double sink = 0.0;
  for (int repeat_index = 1; repeat_index < kComputeRepeats; ++repeat_index) {
    for (auto pixel_value : temp) {
      sink += static_cast<double>(pixel_value) * scale;
    }
  }

  result = temp;
  return result;
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
    final_output.clear();
    final_output.resize(total_size + 2);
    final_output[0] = static_cast<uint8_t>(width);
    final_output[1] = static_cast<uint8_t>(height);
  }

  MPI_Gatherv(local_result.data(), local_size, MPI_UNSIGNED_CHAR, rank == 0 ? final_output.data() + 2 : nullptr,
              recv_counts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(final_output.size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    final_output.resize(total_size);
  }
  MPI_Bcast(final_output.data(), total_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
}

bool ContrastEnhancementMPI::RunImpl() {
  try {
    const auto &input = GetInput();

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = static_cast<int>(input[0]);
    int height = static_cast<int>(input[1]);
    int total_pixels = width * height;

    int base_pixels = total_pixels / size;
    int remainder = total_pixels % size;

    int my_pixels = base_pixels + (rank < remainder ? 1 : 0);
    int offset = 0;
    for (int i = 0; i < rank; ++i) {
      offset += base_pixels + (i < remainder ? 1 : 0);
    }

    std::vector<uint8_t> local_pixels;
    if (my_pixels > 0) {
      local_pixels.reserve(my_pixels);
      for (int i = 0; i < my_pixels; ++i) {
        local_pixels.push_back(input[2 + offset + i]);
      }
    }

    unsigned char global_min = 0;
    unsigned char global_max = 0;
    FindGlobalMinMax(local_pixels, global_min, global_max);

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
  const auto &out = GetOutput();

  if (out.size() < 2) {
    return false;
  }

  int width_out = static_cast<int>(out[0]);
  int height_out = static_cast<int>(out[1]);

  int width_in = static_cast<int>(GetInput()[0]);
  int height_in = static_cast<int>(GetInput()[1]);

  if (width_out != width_in || height_out != height_in) {
    return false;
  }
  if (out.size() != static_cast<size_t>(width_out * height_out) + 2) {
    return false;
  }

  for (auto pixel : out) {
    if (pixel < kMinPixelValue || pixel > kMaxPixelValue) {
      return false;
    }
  }

  return true;
}

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
