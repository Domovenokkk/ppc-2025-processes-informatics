#include "mityaeva_d_contrast_enhancement_histogram_stretching/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

ContrastEnhancementSEQ::ContrastEnhancementSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<uint8_t>{};
}

bool ContrastEnhancementSEQ::ValidationImpl() {
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

bool ContrastEnhancementSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  min_pixel_ = MAX_PIXEL_VALUE;
  max_pixel_ = MIN_PIXEL_VALUE;

  for (size_t i = 2; i < input.size(); ++i) {
    uint8_t pixel = input[i];
    if (pixel < min_pixel_) {
      min_pixel_ = pixel;
    }
    if (pixel > max_pixel_) {
      max_pixel_ = pixel;
    }
  }

  return true;
}

bool ContrastEnhancementSEQ::RunImpl() {
  try {
    const auto &input = GetInput();
    const size_t input_size = input.size();

    // Быстрая проверка
    if (input_size < 2) {
      return false;
    }

    // Создаем результат напрямую, без промежуточного вектора
    OutType result;

    // Добавляем размеры изображения
    result.push_back(static_cast<uint8_t>(width_));
    result.push_back(static_cast<uint8_t>(height_));

    // Если все пиксели одинаковые, просто копируем
    if (min_pixel_ == max_pixel_) {
      // Резервируем память для эффективности
      result.reserve(input_size);

      // Копируем все пиксели, начиная с индекса 2
      for (size_t i = 2; i < input_size; ++i) {
        result.push_back(input[i]);
      }
    } else {
      // Вычисляем коэффициент растяжки
      double scale =
          static_cast<double>(MAX_PIXEL_VALUE - MIN_PIXEL_VALUE) / static_cast<double>(max_pixel_ - min_pixel_);

      // Резервируем память для эффективности
      result.reserve(input_size);

      // Обрабатываем все пиксели
      for (size_t i = 2; i < input_size; ++i) {
        uint8_t pixel = input[i];
        double new_value = static_cast<double>(pixel - min_pixel_) * scale;

        // Округляем и ограничиваем диапазон
        double rounded_value = new_value + 0.5;
        if (rounded_value < 0.0) {
          rounded_value = 0.0;
        }
        if (rounded_value > 255.0) {
          rounded_value = 255.0;
        }

        result.push_back(static_cast<uint8_t>(rounded_value));
      }
    }

    // Устанавливаем результат
    GetOutput().swap(result);
    return true;

  } catch (...) {
    return false;
  }
}

bool ContrastEnhancementSEQ::PostProcessingImpl() {
  const auto &output = GetOutput();

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

  // Проверка диапазона не нужна для uint8_t, так как он всегда 0-255
  // Но можно проверить, что данные вообще есть
  if (output.size() <= 2) {
    return false;
  }

  return true;
}

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
