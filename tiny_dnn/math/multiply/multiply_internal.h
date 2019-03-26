#pragma once

namespace tiny_dnn {
  namespace math {

    template<typename T>
    void multiply_internal(const Matrix<T>& left, const Matrix<T>& right, Matrix<T>& result) {

        for (int row_left = 0; row_left < left.rowCount(); row_left++)
        {
            for (int col_right = 0; col_right < right.colCount(); col_right++)
            {
                T sum = 0;
                for (int i = 0; i < left.colCount(); i++)
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