#include "rychkova_d_image_smoothing/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "rychkova_d_image_smoothing/common/include/common.hpp"

namespace rychkova_d_image_smoothing {

namespace {

void BroadcastMeta(std::size_t *width, std::size_t *height, std::size_t *channels) {
  std::uint64_t w64 = 0;
  std::uint64_t h64 = 0;
  std::uint64_t ch64 = 0;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    w64 = static_cast<std::uint64_t>(*width);
    h64 = static_cast<std::uint64_t>(*height);
    ch64 = static_cast<std::uint64_t>(*channels);
  }

  MPI_Bcast(&w64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&h64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ch64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  *width = static_cast<std::size_t>(w64);
  *height = static_cast<std::size_t>(h64);
  *channels = static_cast<std::size_t>(ch64);
}

int EffectiveSize(int world_size, std::size_t height) {
  const int h_int = static_cast<int>(height);
  return (h_int < world_size) ? h_int : world_size;
}

void BuildCountsDispls(int world_size, int size_eff, std::size_t height, std::size_t row_size, std::vector<int> *counts,
                       std::vector<int> *displs) {
  counts->assign(world_size, 0);
  displs->assign(world_size, 0);

  std::size_t base = 0;
  std::size_t rem = 0;
  if (size_eff > 0) {
    base = height / static_cast<std::size_t>(size_eff);
    rem = height % static_cast<std::size_t>(size_eff);
  }

  std::size_t offset = 0;
  for (int pid = 0; pid < size_eff; ++pid) {
    const auto pid_sz = static_cast<std::size_t>(pid);
    const std::size_t extra = std::cmp_less(pid_sz, rem) ? 1U : 0U;
    const std::size_t rows = base + extra;
    const std::size_t cnt = rows * row_size;

    (*counts)[pid] = static_cast<int>(cnt);
    (*displs)[pid] = static_cast<int>(offset);
    offset += cnt;
  }
}

std::size_t LocalRowsForRank(int rank, int size_eff, std::size_t height) {
  if (rank >= size_eff || size_eff <= 0) {
    return 0;
  }

  const std::size_t base = height / static_cast<std::size_t>(size_eff);
  const std::size_t rem = height % static_cast<std::size_t>(size_eff);
  const auto rank_sz = static_cast<std::size_t>(rank);
  const std::size_t extra = std::cmp_less(rank_sz, rem) ? 1U : 0U;

  return base + extra;
}

void ExchangeHalo(const std::vector<std::uint8_t> &local_in, std::size_t local_rows, std::size_t row_size, int rank,
                  int size_eff, std::vector<std::uint8_t> *halo_top, std::vector<std::uint8_t> *halo_bottom) {
  halo_top->assign(row_size, 0);
  halo_bottom->assign(row_size, 0);

  // Safety: should not happen for rank < size_eff, but avoid UB if it does.
  if (local_rows == 0 || row_size == 0) {
    return;
  }

  if (rank == 0) {
    for (std::size_t ii = 0; ii < row_size; ++ii) {
      (*halo_top)[ii] = local_in[ii];
    }
  } else {
    MPI_Sendrecv(local_in.data(), static_cast<int>(row_size), MPI_UNSIGNED_CHAR, rank - 1, 0, halo_top->data(),
                 static_cast<int>(row_size), MPI_UNSIGNED_CHAR, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  const std::size_t last_off = (local_rows - 1U) * row_size;
  if (rank == size_eff - 1) {
    for (std::size_t ii = 0; ii < row_size; ++ii) {
      (*halo_bottom)[ii] = local_in[last_off + ii];
    }
  } else {
    MPI_Sendrecv(local_in.data() + last_off, static_cast<int>(row_size), MPI_UNSIGNED_CHAR, rank + 1, 1,
                 halo_bottom->data(), static_cast<int>(row_size), MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
  }
}

inline const std::uint8_t *SelectRowPtr(const std::vector<std::uint8_t> &local_in,
                                        const std::vector<std::uint8_t> &halo_top,
                                        const std::vector<std::uint8_t> &halo_bottom, std::size_t local_y,
                                        std::size_t local_rows, std::size_t row_size, int dy) {
  if (dy == -1) {
    return (local_y == 0U) ? halo_top.data() : (local_in.data() + ((local_y - 1U) * row_size));
  }
  if (dy == 0) {
    return local_in.data() + (local_y * row_size);
  }
  return (local_y + 1U == local_rows) ? halo_bottom.data() : (local_in.data() + ((local_y + 1U) * row_size));
}

inline std::uint8_t SmoothPixel(const std::vector<std::uint8_t> &local_in, const std::vector<std::uint8_t> &halo_top,
                                const std::vector<std::uint8_t> &halo_bottom, std::size_t local_y, std::size_t x_pos,
                                std::size_t ch_pos, std::size_t width, std::size_t channels, std::size_t row_size,
                                std::size_t local_rows) {
  int sum = 0;

  for (int dy = -1; dy <= 1; ++dy) {
    const std::uint8_t *row_ptr = SelectRowPtr(local_in, halo_top, halo_bottom, local_y, local_rows, row_size, dy);

    for (int dx = -1; dx <= 1; ++dx) {
      const auto nx =
          std::clamp<std::int64_t>(static_cast<std::int64_t>(x_pos) + dx, 0, static_cast<std::int64_t>(width) - 1);
      const auto ix = static_cast<std::size_t>(nx);
      sum += row_ptr[((ix * channels) + ch_pos)];
    }
  }

  return static_cast<std::uint8_t>(sum / 9);
}

void SmoothLocal(const std::vector<std::uint8_t> &local_in, std::size_t local_rows, std::size_t width,
                 std::size_t channels, std::size_t row_size, const std::vector<std::uint8_t> &halo_top,
                 const std::vector<std::uint8_t> &halo_bottom, std::vector<std::uint8_t> *local_out) {
  local_out->assign(local_rows * row_size, 0);

  for (std::size_t ly = 0; ly < local_rows; ++ly) {
    for (std::size_t xx = 0; xx < width; ++xx) {
      for (std::size_t cc = 0; cc < channels; ++cc) {
        const auto out_idx = (((ly * width) + xx) * channels) + cc;
        (*local_out)[out_idx] =
            SmoothPixel(local_in, halo_top, halo_bottom, ly, xx, cc, width, channels, row_size, local_rows);
      }
    }
  }
}

}  // namespace

ImageSmoothingMPI::ImageSmoothingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool ImageSmoothingMPI::ValidationImpl() {
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
  return in.data.size() == in.width * in.height * in.channels;
}

bool ImageSmoothingMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &in = GetInput();
    auto &out = GetOutput();
    out.width = in.width;
    out.height = in.height;
    out.channels = in.channels;
    out.data.assign(in.data.size(), 0);
  } else {
    GetOutput() = {};
  }
  return true;
}

bool ImageSmoothingMPI::RunImpl() {
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::size_t width = 0;
  std::size_t height = 0;
  std::size_t channels = 0;

  if (rank == 0) {
    width = GetInput().width;
    height = GetInput().height;
    channels = GetInput().channels;
  }

  BroadcastMeta(&width, &height, &channels);
  if (width == 0 || height == 0 || (channels != 1 && channels != 3)) {
    return false;
  }

  const std::size_t row_size = width * channels;
  const int size_eff = EffectiveSize(world_size, height);

  std::vector<int> counts;
  std::vector<int> displs;
  BuildCountsDispls(world_size, size_eff, height, row_size, &counts, &displs);

  const std::size_t local_rows = LocalRowsForRank(rank, size_eff, height);
  const std::size_t local_size = local_rows * row_size;

  std::vector<std::uint8_t> local_in;
  if (local_size > 0) {
    local_in.resize(local_size);
  }

  std::vector<std::uint8_t> local_out;

  const auto *sendbuf = (rank == 0) ? GetInput().data.data() : nullptr;

  MPI_Scatterv(sendbuf, counts.data(), displs.data(), MPI_UNSIGNED_CHAR, local_size ? local_in.data() : nullptr,
               static_cast<int>(local_size), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  if (rank >= size_eff) {
    auto *recvbuf = (rank == 0) ? GetOutput().data.data() : nullptr;
    MPI_Gatherv(nullptr, 0, MPI_UNSIGNED_CHAR, recvbuf, counts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0,
                MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  std::vector<std::uint8_t> halo_top;
  std::vector<std::uint8_t> halo_bottom;
  ExchangeHalo(local_in, local_rows, row_size, rank, size_eff, &halo_top, &halo_bottom);

  SmoothLocal(local_in, local_rows, width, channels, row_size, halo_top, halo_bottom, &local_out);

  auto *recvbuf = (rank == 0) ? GetOutput().data.data() : nullptr;
  MPI_Gatherv(local_out.data(), static_cast<int>(local_out.size()), MPI_UNSIGNED_CHAR, recvbuf, counts.data(),
              displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool ImageSmoothingMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &in = GetInput();
    const auto &out = GetOutput();
    if (out.width != in.width) {
      return false;
    }
    if (out.height != in.height) {
      return false;
    }
    if (out.channels != in.channels) {
      return false;
    }
    if (out.data.size() != in.data.size()) {
      return false;
    }
  }
  return true;
}

}  // namespace rychkova_d_image_smoothing
