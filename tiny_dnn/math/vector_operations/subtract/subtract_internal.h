#pragma once

#include "tiny_dnn/util/types/types.h"

namespace tiny_dnn {
  namespace math {

    void subtract_internal(const tensor_t& left, const tensor_t& right, tensor_t& result) {

        /*
        for (unsigned i = 0; i < left.getElementCount(); i++) {
            result[i] = left[i] - right[i];
        }
        */
    }

  }
}