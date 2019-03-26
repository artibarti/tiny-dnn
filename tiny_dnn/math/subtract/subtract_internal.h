#pragma once

namespace tiny_dnn {
  namespace math {

    template<typename T>
    void subtract_internal(const Matrix<T>& left, const Matrix<T>& right, Matrix<T>& result) {

        for (unsigned i = 0; i < left.getElementCount(); i++) {
            result[i] = left[i] - right[i];
        }
    }

  }
}