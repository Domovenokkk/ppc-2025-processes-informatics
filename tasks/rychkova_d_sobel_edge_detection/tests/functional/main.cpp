#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "rychkova_d_sobel_edge_detection/common/include/common.hpp"
#include "rychkova_d_sobel_edge_detection/mpi/include/ops_mpi.hpp"
#include "rychkova_d_sobel_edge_detection/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace rychkova_d_sobel_edge_detection {

class RychkovaDRunFuncTestsSobel : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &img = std::get<0>(test_param);
    return std::to_string(img.width) + "x" + std::to_string(img.height) + "_ch" + std::to_string(img.channels) + "_" +
           std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_ = ReferenceSobelAbsSumDiv4(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.data.empty()) {
      return true;
    }

    if (output_data.width != expected_.width) {
      return false;
    }
    if (output_data.height != expected_.height) {
      return false;
    }
    if (output_data.channels != expected_.channels) {
      return false;
    }
    if (output_data.data.size() != expected_.data.size()) {
      return false;
    }

    return output_data.data == expected_.data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  static Image MakeConst(std::size_t w, std::size_t h, std::size_t ch, std::uint8_t v) {
    Image img;
    img.width = w;
    img.height = h;
    img.channels = ch;
    img.data.assign(w * h * ch, v);
    return img;
  }

  static Image MakePattern(std::size_t w, std::size_t h, std::size_t ch) {
    Image img;
    img.width = w;
    img.height = h;
    img.channels = ch;
    img.data.resize(w * h * ch);

    for (std::size_t yy = 0; yy < h; ++yy) {
      for (std::size_t xx = 0; xx < w; ++xx) {
        for (std::size_t cc = 0; cc < ch; ++cc) {
          const auto idx = (((yy * w) + xx) * ch) + cc;
          img.data[idx] = static_cast<std::uint8_t>((idx * 37 + 13) % 256);
        }
      }
    }
    return img;
  }

  static Image ToGray(const Image &in) {
    if (in.channels == 1) {
      return in;
    }

    Image g;
    g.width = in.width;
    g.height = in.height;
    g.channels = 1;
    g.data.assign(in.width * in.height, 0);

    const std::size_t pixels = in.width * in.height;
    for (std::size_t idx_px = 0; idx_px < pixels; ++idx_px) {
      const std::size_t base = (idx_px * 3U);
      const std::uint8_t r = in.data[base + 0U];
      const std::uint8_t gg = in.data[base + 1U];
      const std::uint8_t b = in.data[base + 2U];
      const int y = (77 * r + 150 * gg + 29 * b) >> 8;
      g.data[idx_px] = static_cast<std::uint8_t>(y);
    }
    return g;
  }

  static Image ReferenceSobelAbsSumDiv4(const Image &in_any) {
    const Image in = ToGray(in_any);

    Image out;
    out.width = in.width;
    out.height = in.height;
    out.channels = 1;
    out.data.assign(in.width * in.height, 0);

    const std::size_t w = in.width;
    const std::size_t h = in.height;

    if (w < 3 || h < 3) {
      return out;
    }

    auto idx = [w](std::size_t col, std::size_t row) { return (row * w) + col; };

    for (std::size_t row = 1; (row + 1U) < h; ++row) {
      for (std::size_t col = 1; (col + 1U) < w; ++col) {
        const int p00 = static_cast<int>(in.data[idx(col - 1U, row - 1U)]);
        const int p10 = static_cast<int>(in.data[idx(col, row - 1U)]);
        const int p20 = static_cast<int>(in.data[idx(col + 1U, row - 1U)]);

        const int p01 = static_cast<int>(in.data[idx(col - 1U, row)]);
        const int p21 = static_cast<int>(in.data[idx(col + 1U, row)]);

        const int p02 = static_cast<int>(in.data[idx(col - 1U, row + 1U)]);
        const int p12 = static_cast<int>(in.data[idx(col, row + 1U)]);
        const int p22 = static_cast<int>(in.data[idx(col + 1U, row + 1U)]);

        const int gx = (-p00 + p20) + (-2 * p01 + 2 * p21) + (-p02 + p22);
        const int gy = (-p00 - 2 * p10 - p20) + (p02 + 2 * p12 + p22);

        int mag = std::abs(gx) + std::abs(gy);
        mag /= 4;

        mag = std::max(mag, 0);
        mag = std::min(mag, 255);

        out.data[idx(col, row)] = static_cast<std::uint8_t>(mag);
      }
    }

    return out;
  }

  InType input_data_{};
  OutType expected_{};

 public:
  static TestType ParamConst(std::size_t w, std::size_t h, std::size_t ch, std::uint8_t v, const std::string &name) {
    return std::make_tuple(MakeConst(w, h, ch, v), name);
  }

  static TestType ParamPattern(std::size_t w, std::size_t h, std::size_t ch, const std::string &name) {
    return std::make_tuple(MakePattern(w, h, ch), name);
  }
};

namespace {

TEST_P(RychkovaDRunFuncTestsSobel, SobelFromGeneratedImage) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    RychkovaDRunFuncTestsSobel::ParamPattern(2, 2, 1, "gray_2x2_pattern"),
    RychkovaDRunFuncTestsSobel::ParamConst(8, 6, 1, 128, "gray_const_8x6_128"),
    RychkovaDRunFuncTestsSobel::ParamPattern(19, 11, 1, "gray_19x11_pattern"),
    RychkovaDRunFuncTestsSobel::ParamPattern(16, 9, 3, "rgb_16x9_pattern"),
    RychkovaDRunFuncTestsSobel::ParamPattern(32, 32, 1, "gray_32x32_pattern"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SobelEdgeDetectionMPI, InType>(kTestParam, PPC_SETTINGS_rychkova_d_sobel_edge_detection),
    ppc::util::AddFuncTask<SobelEdgeDetectionSEQ, InType>(kTestParam, PPC_SETTINGS_rychkova_d_sobel_edge_detection));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = RychkovaDRunFuncTestsSobel::PrintFuncTestName<RychkovaDRunFuncTestsSobel>;

INSTANTIATE_TEST_SUITE_P(SobelEdgeDetectionTests, RychkovaDRunFuncTestsSobel, kGtestValues, kTestName);

}  // namespace

}  // namespace rychkova_d_sobel_edge_detection
