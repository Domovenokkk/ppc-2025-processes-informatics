#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>

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
  static Image MakeConst(std::size_t width, std::size_t height, std::size_t channels, std::uint8_t value) {
    Image img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.data.assign(width * height * channels, value);
    return img;
  }

  static Image MakePattern(std::size_t width, std::size_t height, std::size_t channels) {
    Image img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.data.resize(width * height * channels);

    for (std::size_t yy = 0; yy < height; ++yy) {
      for (std::size_t xx = 0; xx < width; ++xx) {
        for (std::size_t cc = 0; cc < channels; ++cc) {
          const auto idx = (((yy * width) + xx) * channels) + cc;
          img.data[idx] = static_cast<std::uint8_t>((((idx * 37U) + 13U) % 256U));
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

    const std::size_t width = in.width;
    const std::size_t height = in.height;
    const std::size_t channels = in.channels;

    auto clamp_i64 = [](std::int64_t value, std::int64_t lo, std::int64_t hi) {
      if (value < lo) {
        return lo;
      }
      if (value > hi) {
        return hi;
      }
      return value;
    };

    for (std::size_t yy = 0; yy < height; ++yy) {
      for (std::size_t xx = 0; xx < width; ++xx) {
        for (std::size_t cc = 0; cc < channels; ++cc) {
          int sum = 0;

          for (int dy = -1; dy <= 1; ++dy) {
            const auto ny = clamp_i64(static_cast<std::int64_t>(yy) + dy, 0, static_cast<std::int64_t>(height) - 1);

            for (int dx = -1; dx <= 1; ++dx) {
              const auto nx = clamp_i64(static_cast<std::int64_t>(xx) + dx, 0, static_cast<std::int64_t>(width) - 1);

              const auto ix = static_cast<std::size_t>(nx);
              const auto iy = static_cast<std::size_t>(ny);
              const auto idx = (((iy * width) + ix) * channels) + cc;

              sum += static_cast<int>(in.data[idx]);
            }
          }

          const auto out_idx = (((yy * width) + xx) * channels) + cc;
          out.data[out_idx] = static_cast<std::uint8_t>(sum / 9);
        }
      }
    }

    return out;
  }

  InType input_data_{};
  OutType expected_{};

 public:
  static TestType ParamConst(std::size_t width, std::size_t height, std::size_t channels, std::uint8_t value,
                             const std::string &name) {
    return std::make_tuple(MakeConst(width, height, channels, value), name);
  }

  static TestType ParamPattern(std::size_t width, std::size_t height, std::size_t channels, const std::string &name) {
    return std::make_tuple(MakePattern(width, height, channels), name);
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
