#pragma once

namespace tiny_dnn {
  namespace math {

    template<typename T>
    void add_internal(const matrix_t& left, const matrix_t& right, matrix_t& result) {

        for (unsigned i = 0; i < left.getElementCount(); i++) {
            result[i] = left[i] + right[i];
        }
    }

  }
}