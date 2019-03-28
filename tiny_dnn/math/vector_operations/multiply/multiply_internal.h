#pragma once

#include "tiny_dnn/util/types/types.h"

namespace tiny_dnn {
  namespace math {

    void multiply_internal(const matrix_t& left, const matrix_t& right, matrix_t& result) {

        for (unsigned row_left = 0; row_left < left.rowCount(); row_left++)
        {
            for (unsigned col_right = 0; col_right < right.colCount(); col_right++)
            {
                float_t sum = 0;
                for (unsigned i = 0; i < left.colCount(); i++)
                {
                    sum += left[row_left * left.colCount() + i] 
                        * right[i * right.colCount() + col_right];
                }
                result[row_left * result.colCount() + col_right] = sum;
            }
        }

    }

  }
}