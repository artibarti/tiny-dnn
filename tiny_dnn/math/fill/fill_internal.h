#pragma once


namespace tiny_dnn {
  namespace math {

    template<typename T>
    void fill_internal(Matrix<T>& mat, const T& value = 0) {
        
        for(unsigned i = 0; i < mat.getElementCount(); i++) {
            mat[i] = value;
        }
    }

  }
}