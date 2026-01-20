#include "rychkova_d_sobel_edge_detection/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "rychkova_d_sobel_edge_detection/common/include/common.hpp"

namespace rychkova_d_sobel_edge_detection {

namespace {

inline uint8_t ClampToU8(int v) {
  if (v < 0) {
    return 0;
  }
  if (v > 255) {
    return 255;
  }
  return static_cast<uint8_t>(v);
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
      std::copy(in.data.begin(), in.data.end(), gray_.begin());
    } else {
      for (std::size_t i = 0; i < pixels; ++i) {
        const uint8_t r = in.data[i * 3 + 0];
        const uint8_t g = in.data[i * 3 + 1];
        const uint8_t b = in.data[i * 3 + 2];
        const int y = (77 * r + 150 * g + 29 * b) >> 8;
        gray_[i] = static_cast<uint8_t>(y);
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

  std::size_t w = 0;
  std::size_t h = 0;

  if (rank == 0) {
    w = GetInput().width;
    h = GetInput().height;
  }

  MPI_Bcast(&w, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&h, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  if (w == 0 || h == 0) {
    return false;
  }

  if (w < 3 || h < 3) {
    if (rank == 0) {
      std::fill(out_data_.begin(), out_data_.end(), 0);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  const std::size_t base = h / static_cast<std::size_t>(size);
  const std::size_t rem = h % static_cast<std::size_t>(size);

  const std::size_t local_rows = base + (static_cast<std::size_t>(rank) < rem ? 1 : 0);

  const std::size_t start_row =
      base * static_cast<std::size_t>(rank) + std::min<std::size_t>(static_cast<std::size_t>(rank), rem);

  const bool has_top = (start_row > 0);
  const bool has_bottom = (start_row + local_rows < h);
  const std::size_t halo_top = has_top ? 1 : 0;
  const std::size_t halo_bottom = has_bottom ? 1 : 0;

  const std::size_t recv_rows = local_rows + halo_top + halo_bottom;
  const std::size_t recv_count = recv_rows * w;

  std::vector<uint8_t> gray_chunk(recv_count, 0);

  std::vector<int> sendcounts;
  std::vector<int> displs;
  if (rank == 0) {
    sendcounts.resize(size, 0);
    displs.resize(size, 0);

    for (int r = 0; r < size; ++r) {
      const std::size_t lr = base + (static_cast<std::size_t>(r) < rem ? 1 : 0);
      const std::size_t sr =
          base * static_cast<std::size_t>(r) + std::min<std::size_t>(static_cast<std::size_t>(r), rem);

      const bool top = (sr > 0);
      const bool bottom = (sr + lr < h);
      const std::size_t ht = top ? 1 : 0;
      const std::size_t hb = bottom ? 1 : 0;

      const std::size_t rr = lr + ht + hb;
      const std::size_t count = rr * w;
      const std::size_t disp_row = sr - ht;
      const std::size_t disp = disp_row * w;

      sendcounts[r] = static_cast<int>(count);
      displs[r] = static_cast<int>(disp);
    }
  }

  MPI_Scatterv(rank == 0 ? gray_.data() : nullptr, rank == 0 ? sendcounts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, MPI_UNSIGNED_CHAR, gray_chunk.data(), static_cast<int>(recv_count),
               MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  std::vector<uint8_t> local_out(local_rows * w, 0);

  auto idx_local = [w](std::size_t x, std::size_t y) { return y * w + x; };

  for (std::size_t y = 0; y < local_rows; ++y) {
    const std::size_t global_y = start_row + y;

    if (global_y == 0 || global_y + 1 == h) {
      std::fill(local_out.begin() + y * w, local_out.begin() + (y + 1) * w, 0);
      continue;
    }

    const std::size_t cy = y + halo_top;

    local_out[idx_local(0, y)] = 0;
    local_out[idx_local(w - 1, y)] = 0;

    for (std::size_t x = 1; x + 1 < w; ++x) {
      const int p00 = static_cast<int>(gray_chunk[idx_local(x - 1, cy - 1)]);
      const int p10 = static_cast<int>(gray_chunk[idx_local(x, cy - 1)]);
      const int p20 = static_cast<int>(gray_chunk[idx_local(x + 1, cy - 1)]);

      const int p01 = static_cast<int>(gray_chunk[idx_local(x - 1, cy)]);
      const int p21 = static_cast<int>(gray_chunk[idx_local(x + 1, cy)]);

      const int p02 = static_cast<int>(gray_chunk[idx_local(x - 1, cy + 1)]);
      const int p12 = static_cast<int>(gray_chunk[idx_local(x, cy + 1)]);
      const int p22 = static_cast<int>(gray_chunk[idx_local(x + 1, cy + 1)]);

      const int gx = (-p00 + p20) + (-2 * p01 + 2 * p21) + (-p02 + p22);
      const int gy = (-p00 - 2 * p10 - p20) + (p02 + 2 * p12 + p22);

      int mag = std::abs(gx) + std::abs(gy);
      mag /= 4;

      local_out[idx_local(x, y)] = ClampToU8(mag);
    }
  }

  std::vector<int> recvcounts_out;
  std::vector<int> displs_out;
  if (rank == 0) {
    recvcounts_out.resize(size, 0);
    displs_out.resize(size, 0);

    for (int r = 0; r < size; ++r) {
      const std::size_t lr = base + (static_cast<std::size_t>(r) < rem ? 1 : 0);
      const std::size_t sr =
          base * static_cast<std::size_t>(r) + std::min<std::size_t>(static_cast<std::size_t>(r), rem);

      recvcounts_out[r] = static_cast<int>(lr * w);
      displs_out[r] = static_cast<int>(sr * w);
    }
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
