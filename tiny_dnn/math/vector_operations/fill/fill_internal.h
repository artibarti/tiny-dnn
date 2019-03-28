#pragma once

#include "tiny_dnn/util/types/types.h"

namespace tiny_dnn {
  namespace math {

    void fill_internal(matrix_t& mat, const float_t& value = 0) {
        
        for(unsigned i = 0; i < mat.getElementCount(); i++) {
            mat[i] = value;
        }
    }

  }
}