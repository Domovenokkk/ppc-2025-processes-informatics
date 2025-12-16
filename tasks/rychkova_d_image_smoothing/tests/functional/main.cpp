#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "rychkova_d_image_smoothing/common/include/common.hpp"
#include "rychkova_d_image_smoothing/mpi/include/ops_mpi.hpp"
#include "rychkova_d_image_smoothing/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace rychkova_d_image_smoothing {

class RychkovaDRunFuncTestsImageSmoothing : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
    expected_ = ReferenceSmooth3x3Clamp(input_data_);
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
          const auto idx = (((yy * w) + xx) * ch) + cc;  // FIX 2
          img.data[idx] = static_cast<std::uint8_t>((idx * 37 + 13) % 256);
        }
      }
    }
    return img;
  }

  static Image ReferenceSmooth3x3Clamp(const Image &in) {
    Image out;
    out.width = in.width;
    out.height = in.height;
    out.channels = in.channels;
    out.data.assign(in.data.size(), 0);

    const std::size_t w = in.width;
    const std::size_t h = in.height;
    const std::size_t ch = in.channels;

    auto clamp_i64 = [](std::int64_t v, std::int64_t lo, std::int64_t hi) {
      if (v < lo) {
        return lo;
      }
      if (v > hi) {
        return hi;
      }
      return v;
    };

    for (std::size_t yy = 0; yy < h; ++yy) {
      for (std::size_t xx = 0; xx < w; ++xx) {
        for (std::size_t cc = 0; cc < ch; ++cc) {
          int sum = 0;

          for (int dy = -1; dy <= 1; ++dy) {
            const auto ny = clamp_i64(static_cast<std::int64_t>(yy) + dy, 0, static_cast<std::int64_t>(h) - 1);

            for (int dx = -1; dx <= 1; ++dx) {
              const auto nx = clamp_i64(static_cast<std::int64_t>(xx) + dx, 0, static_cast<std::int64_t>(w) - 1);

              const auto ix = static_cast<std::size_t>(nx);
              const auto iy = static_cast<std::size_t>(ny);
              const auto idx = (((iy * w) + ix) * ch) + cc;  // FIX 2

              sum += static_cast<int>(in.data[idx]);
            }
          }

          const auto out_idx = (((yy * w) + xx) * ch) + cc;  // FIX 2
          out.data[out_idx] = static_cast<std::uint8_t>(sum / 9);
        }
      }
    }
    return out;
  }

 private:  // FIX 3: только один private
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

TEST_P(RychkovaDRunFuncTestsImageSmoothing, SmoothingFromGeneratedImage) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
    RychkovaDRunFuncTestsImageSmoothing::ParamPattern(2, 2, 1, "gray_2x2_pattern"),
    RychkovaDRunFuncTestsImageSmoothing::ParamConst(8, 6, 1, 128, "gray_const_8x6_128"),
    RychkovaDRunFuncTestsImageSmoothing::ParamPattern(19, 11, 1, "gray_19x11_pattern"),
    RychkovaDRunFuncTestsImageSmoothing::ParamPattern(16, 9, 3, "rgb_16x9_pattern"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ImageSmoothingMPI, InType>(kTestParam, PPC_SETTINGS_rychkova_d_image_smoothing),
    ppc::util::AddFuncTask<ImageSmoothingSEQ, InType>(kTestParam, PPC_SETTINGS_rychkova_d_image_smoothing));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = RychkovaDRunFuncTestsImageSmoothing::PrintFuncTestName<RychkovaDRunFuncTestsImageSmoothing>;

INSTANTIATE_TEST_SUITE_P(ImageSmoothingTests, RychkovaDRunFuncTestsImageSmoothing, kGtestValues, kTestName);

}  // namespace
}  // namespace rychkova_d_image_smoothing
