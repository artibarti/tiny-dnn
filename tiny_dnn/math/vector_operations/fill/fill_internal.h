#pragma once

#include "tiny_dnn/util/types/types.h"

namespace tiny_dnn {
  namespace math {

    void fill_internal(tensor_t& tensor, const float_t& value = 0) {
        
      for(unsigned i = 0; i<tensor.size(); i++) {
        for (unsigned j = 0; j<tensor[i].size(); j++) {
          tensor[i][j] = value;
        }
      }

  }
}
}