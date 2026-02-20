#include "rychkova_d_sobel_edge_detection/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#include "rychkova_d_sobel_edge_detection/common/include/common.hpp"

namespace rychkova_d_sobel_edge_detection {

namespace {

inline std::uint8_t ClampToU8(int value) {
  if (value < 0) {
    return 0;
  }
  if (value > 255) {
    return 255;
  }
  return static_cast<std::uint8_t>(value);
}

struct SplitInfo {
  std::size_t local_rows{};
  std::size_t start_row{};
  std::size_t halo_top{};
  std::size_t halo_bottom{};
};

inline std::uint64_t ToU64(std::size_t value) {
  return static_cast<std::uint64_t>(value);
}

inline std::size_t ToSizeT(std::uint64_t value) {
  return static_cast<std::size_t>(value);
}

SplitInfo ComputeSplit(std::size_t height, int comm_size, int comm_rank) {
  const std::size_t base = height / static_cast<std::size_t>(comm_size);
  const std::size_t rem = height % static_cast<std::size_t>(comm_size);

  const auto rank_sz = static_cast<std::size_t>(comm_rank);

  SplitInfo info{};
  info.local_rows = base + (std::cmp_less(rank_sz, rem) ? 1U : 0U);
  info.start_row = (base * rank_sz) + std::min<std::size_t>(rank_sz, rem);

  const bool has_top = (info.start_row > 0);
  const bool has_bottom = (info.start_row + info.local_rows < height);

  info.halo_top = has_top ? 1U : 0U;
  info.halo_bottom = has_bottom ? 1U : 0U;
  return info;
}

std::pair<std::vector<int>, std::vector<int>> BuildScattervLayout(std::size_t width, std::size_t height,
                                                                  int comm_size) {
  std::vector<int> sendcounts(comm_size, 0);
  std::vector<int> displs(comm_size, 0);

  const std::size_t base = height / static_cast<std::size_t>(comm_size);
  const std::size_t rem = height % static_cast<std::size_t>(comm_size);

  for (int rank_i = 0; rank_i < comm_size; ++rank_i) {
    const auto rank_sz = static_cast<std::size_t>(rank_i);

    const std::size_t local_rows = base + (std::cmp_less(rank_sz, rem) ? 1U : 0U);
    const std::size_t start_row = (base * rank_sz) + std::min<std::size_t>(rank_sz, rem);

    const bool has_top = (start_row > 0);
    const bool has_bottom = (start_row + local_rows < height);

    const std::size_t halo_top = has_top ? 1U : 0U;
    const std::size_t halo_bottom = has_bottom ? 1U : 0U;

    const std::size_t recv_rows = local_rows + halo_top + halo_bottom;
    const std::size_t count = recv_rows * width;

    const std::size_t disp_row = start_row - halo_top;
    const std::size_t disp = disp_row * width;

    sendcounts[rank_i] = static_cast<int>(count);
    displs[rank_i] = static_cast<int>(disp);
  }

  return {std::move(sendcounts), std::move(displs)};
}

std::pair<std::vector<int>, std::vector<int>> BuildGathervLayout(std::size_t width, std::size_t height, int comm_size) {
  std::vector<int> recvcounts(comm_size, 0);
  std::vector<int> displs(comm_size, 0);

  const std::size_t base = height / static_cast<std::size_t>(comm_size);
  const std::size_t rem = height % static_cast<std::size_t>(comm_size);

  for (int rank_i = 0; rank_i < comm_size; ++rank_i) {
    const auto rank_sz = static_cast<std::size_t>(rank_i);

    const std::size_t local_rows = base + (std::cmp_less(rank_sz, rem) ? 1U : 0U);
    const std::size_t start_row = (base * rank_sz) + std::min<std::size_t>(rank_sz, rem);

    recvcounts[rank_i] = static_cast<int>(local_rows * width);
    displs[rank_i] = static_cast<int>(start_row * width);
  }

  return {std::move(recvcounts), std::move(displs)};
}

void ComputeSobelChunk(const std::vector<std::uint8_t> &gray_chunk, std::vector<std::uint8_t> *local_out,
                       std::size_t width, std::size_t height, const SplitInfo &split) {
  auto idx = [width](std::size_t col, std::size_t row) { return (row * width) + col; };

  for (std::size_t row = 0; row < split.local_rows; ++row) {
    const std::size_t global_row = split.start_row + row;

    // Границы изображения -> нули
    if (global_row == 0 || (global_row + 1U) == height) {
      const std::size_t offset = row * width;
      std::ranges::fill(std::span<std::uint8_t>(*local_out).subspan(offset, width), 0);
      continue;
    }

    const std::size_t cy = row + split.halo_top;

    (*local_out)[idx(0, row)] = 0;
    (*local_out)[idx(width - 1U, row)] = 0;

    for (std::size_t col = 1; (col + 1U) < width; ++col) {
      const int p00 = static_cast<int>(gray_chunk[idx(col - 1U, cy - 1U)]);
      const int p10 = static_cast<int>(gray_chunk[idx(col, cy - 1U)]);
      const int p20 = static_cast<int>(gray_chunk[idx(col + 1U, cy - 1U)]);

      const int p01 = static_cast<int>(gray_chunk[idx(col - 1U, cy)]);
      const int p21 = static_cast<int>(gray_chunk[idx(col + 1U, cy)]);

      const int p02 = static_cast<int>(gray_chunk[idx(col - 1U, cy + 1U)]);
      const int p12 = static_cast<int>(gray_chunk[idx(col, cy + 1U)]);
      const int p22 = static_cast<int>(gray_chunk[idx(col + 1U, cy + 1U)]);

      const int gx = (-p00 + p20) + (-2 * p01 + 2 * p21) + (-p02 + p22);
      const int gy = (-p00 - 2 * p10 - p20) + (p02 + 2 * p12 + p22);

      int mag = std::abs(gx) + std::abs(gy);
      mag /= 4;

      (*local_out)[idx(col, row)] = ClampToU8(mag);
    }
  }
}

}  // namespace

SobelEdgeDetectionMPI::SobelEdgeDetectionMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool SobelEdgeDetectionMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    return true;
  }

  const auto &in = GetInput();
  if (in.width == 0 || in.height == 0) {
    return false;
  }
  if (in.channels != 1 && in.channels != 3) {
    return false;
  }

  const std::size_t expected = in.width * in.height * in.channels;
  if (in.data.size() != expected) {
    return false;
  }

  const auto &out = GetOutput();
  return out.data.empty() && out.width == 0 && out.height == 0;
}

bool SobelEdgeDetectionMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &in = GetInput();
    auto &out = GetOutput();

    out.width = in.width;
    out.height = in.height;
    out.channels = 1;
    out.data.clear();

    out_data_.assign(in.width * in.height, 0);

    const std::size_t pixels = in.width * in.height;
    gray_.assign(pixels, 0);

    if (in.channels == 1) {
      std::ranges::copy(in.data, gray_.begin());
    } else {
      for (std::size_t idx_px = 0; idx_px < pixels; ++idx_px) {
        const std::size_t base = (idx_px * 3U);
        const std::uint8_t r = in.data[base + 0U];
        const std::uint8_t g = in.data[base + 1U];
        const std::uint8_t b = in.data[base + 2U];
        const int y = (77 * r + 150 * g + 29 * b) >> 8;
        gray_[idx_px] = static_cast<std::uint8_t>(y);
      }
    }
  }

  return true;
}

bool SobelEdgeDetectionMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::uint64_t w_u64 = 0;
  std::uint64_t h_u64 = 0;

  if (rank == 0) {
    w_u64 = ToU64(GetInput().width);
    h_u64 = ToU64(GetInput().height);
  }

  MPI_Bcast(&w_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&h_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  const std::size_t width = ToSizeT(w_u64);
  const std::size_t height = ToSizeT(h_u64);

  if (width == 0 || height == 0) {
    return false;
  }

  if (width < 3 || height < 3) {
    if (rank == 0) {
      std::ranges::fill(out_data_, 0);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  const SplitInfo split = ComputeSplit(height, size, rank);

  const std::size_t recv_rows = split.local_rows + split.halo_top + split.halo_bottom;
  const std::size_t recv_count = recv_rows * width;

  std::vector<std::uint8_t> gray_chunk(recv_count, 0);

  std::vector<int> sendcounts;
  std::vector<int> displs;
  if (rank == 0) {
    auto layout = BuildScattervLayout(width, height, size);
    sendcounts = std::move(layout.first);
    displs = std::move(layout.second);
  }

  MPI_Scatterv(rank == 0 ? gray_.data() : nullptr, rank == 0 ? sendcounts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, MPI_UNSIGNED_CHAR, gray_chunk.data(), static_cast<int>(recv_count),
               MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  std::vector<std::uint8_t> local_out(split.local_rows * width, 0);
  ComputeSobelChunk(gray_chunk, &local_out, width, height, split);

  std::vector<int> recvcounts_out;
  std::vector<int> displs_out;
  if (rank == 0) {
    auto layout = BuildGathervLayout(width, height, size);
    recvcounts_out = std::move(layout.first);
    displs_out = std::move(layout.second);
  }

  MPI_Gatherv(local_out.data(), static_cast<int>(local_out.size()), MPI_UNSIGNED_CHAR,
              rank == 0 ? out_data_.data() : nullptr, rank == 0 ? recvcounts_out.data() : nullptr,
              rank == 0 ? displs_out.data() : nullptr, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool SobelEdgeDetectionMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    auto &out = GetOutput();
    out.data = out_data_;
    return (out.data.size() == out.width * out.height * out.channels);
  }

  return true;
}

}  // namespace rychkova_d_sobel_edge_detection
