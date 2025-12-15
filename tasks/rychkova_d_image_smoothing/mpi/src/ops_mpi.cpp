#include "rychkova_d_image_smoothing/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rychkova_d_image_smoothing/common/include/common.hpp"

namespace rychkova_d_image_smoothing {

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

  const std::size_t expected = in.width * in.height * in.channels;
  if (in.data.size() != expected) {
    return false;
  }

  return true;
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
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::size_t w = 0, h = 0, ch = 0;
  if (rank == 0) {
    w = GetInput().width;
    h = GetInput().height;
    ch = GetInput().channels;
  }

  unsigned long long uw = static_cast<unsigned long long>(w);
  unsigned long long uh = static_cast<unsigned long long>(h);
  unsigned long long uch = static_cast<unsigned long long>(ch);

  MPI_Bcast(&uw, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&uh, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&uch, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  w = static_cast<std::size_t>(uw);
  h = static_cast<std::size_t>(uh);
  ch = static_cast<std::size_t>(uch);

  if (w == 0 || h == 0 || (ch != 1 && ch != 3)) {
    return false;
  }

  const std::size_t row_size = w * ch;

  const int h_int = static_cast<int>(h);
  const int size_eff = (h_int < size) ? h_int : size;

  std::vector<int> counts(size, 0);
  std::vector<int> displs(size, 0);

  std::size_t base = 0, rem = 0;
  if (size_eff > 0) {
    base = h / static_cast<std::size_t>(size_eff);
    rem = h % static_cast<std::size_t>(size_eff);
  }

  std::size_t offset = 0;
  for (int r = 0; r < size_eff; ++r) {
    std::size_t rows_r = base + ((static_cast<std::size_t>(r) < rem) ? 1 : 0);
    std::size_t cnt = rows_r * row_size;
    counts[r] = static_cast<int>(cnt);
    displs[r] = static_cast<int>(offset);
    offset += cnt;
  }

  std::size_t local_rows = 0;
  if (rank < size_eff) {
    local_rows = base + ((static_cast<std::size_t>(rank) < rem) ? 1 : 0);
  }

  std::vector<uint8_t> local_in(local_rows * row_size);
  std::vector<uint8_t> local_out(local_rows * row_size);

  const uint8_t *sendbuf = (rank == 0) ? GetInput().data.data() : nullptr;

  MPI_Scatterv(sendbuf, counts.data(), displs.data(), MPI_UNSIGNED_CHAR, (local_in.empty() ? nullptr : local_in.data()),
               static_cast<int>(local_in.size()), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  if (rank >= size_eff) {
    uint8_t *recvbuf = (rank == 0) ? GetOutput().data.data() : nullptr;
    MPI_Gatherv(nullptr, 0, MPI_UNSIGNED_CHAR, recvbuf, counts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0,
                MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  std::vector<uint8_t> halo_top(row_size);
  std::vector<uint8_t> halo_bottom(row_size);

  if (rank == 0) {
    for (std::size_t i = 0; i < row_size; ++i) {
      halo_top[i] = local_in[i];
    }
  } else {
    MPI_Sendrecv(local_in.data(), static_cast<int>(row_size), MPI_UNSIGNED_CHAR, rank - 1, 0, halo_top.data(),
                 static_cast<int>(row_size), MPI_UNSIGNED_CHAR, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (rank == size_eff - 1) {
    const std::size_t last_off = (local_rows - 1) * row_size;
    for (std::size_t i = 0; i < row_size; ++i) {
      halo_bottom[i] = local_in[last_off + i];
    }
  } else {
    const std::size_t last_off = (local_rows - 1) * row_size;
    MPI_Sendrecv(local_in.data() + last_off, static_cast<int>(row_size), MPI_UNSIGNED_CHAR, rank + 1, 1,
                 halo_bottom.data(), static_cast<int>(row_size), MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
  }

  for (std::size_t ly = 0; ly < local_rows; ++ly) {
    const std::size_t cur_off = ly * row_size;

    for (std::size_t x = 0; x < w; ++x) {
      for (std::size_t c = 0; c < ch; ++c) {
        int sum = 0;

        for (int dy = -1; dy <= 1; ++dy) {
          const uint8_t *row_ptr = nullptr;

          if (dy == -1) {
            row_ptr = (ly == 0) ? halo_top.data() : (local_in.data() + (ly - 1) * row_size);
          } else if (dy == 0) {
            row_ptr = local_in.data() + cur_off;
          } else {
            row_ptr = (ly + 1 == local_rows) ? halo_bottom.data() : (local_in.data() + (ly + 1) * row_size);
          }

          for (int dx = -1; dx <= 1; ++dx) {
            long long nx = static_cast<long long>(x) + dx;
            if (nx < 0) {
              nx = 0;
            }
            if (nx >= static_cast<long long>(w)) {
              nx = static_cast<long long>(w) - 1;
            }

            const std::size_t ix = static_cast<std::size_t>(nx);
            sum += row_ptr[ix * ch + c];
          }
        }

        local_out[cur_off + x * ch + c] = static_cast<uint8_t>(sum / 9);
      }
    }
  }

  uint8_t *recvbuf = (rank == 0) ? GetOutput().data.data() : nullptr;

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
