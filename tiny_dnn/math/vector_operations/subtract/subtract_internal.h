#pragma once

#include "tiny_dnn/util/types/types.h"

namespace tiny_dnn {
  namespace math {

    void subtract_internal(const tensor_t& left, const tensor_t& right, tensor_t& result) {

      for (unsigned i = 0; i < left.size(); i++)
        for(unsigned j = 0; j < left[i].size(); j++)
          result[i][j] = left[i][j] - right[i][j];
    }

  }
}