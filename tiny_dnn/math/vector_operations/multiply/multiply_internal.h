#pragma once

#include "tiny_dnn/util/types/types.h"

namespace tiny_dnn {
  namespace math {

    void multiply_internal(const tensor_t& left, const tensor_t& right, tensor_t& result) {

        shape2d leftDim = getDimension(left);
        shape2d rightDim = getDimension(right);

        for (unsigned row_left = 0; row_left < leftDim.x; row_left++)
        {
            for (unsigned col_right = 0; col_right < rightDim.y; col_right++)
            {
                float_t sum = 0;
                for (unsigned i = 0; i < leftDim.y; i++)
                {
                    sum += left[row_left][i] * right[i][col_right];
                }
                result[row_left][col_right] = sum;
            }
        }

    }

  }
}