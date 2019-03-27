#pragma once

#include <vector>
#include <iostream>

#include "tiny_dnn/containers/containers.h"
#include "tiny_dnn/util/aligned_allocator.h"
#include "tiny_dnn/config/config.h"

namespace tiny_dnn {

typedef std::size_t label_t;
typedef std::size_t layer_size_t;  // for backward compatibility

typedef std::vector<float_t, aligned_allocator<float_t, 64>> vec_t;
typedef std::vector<vec_t> tensor_t;

using matrix_t = tiny_dnn::Matrix<float_t>;

// TODO remove after switched to matrix class instead of tensor_t
inline void fill_tensor(tensor_t &tensor, float_t value) {
  for (auto &t : tensor) {
    vectorize::fill(&t[0], t.size(), value);
  }
}

inline void fill_tensor(tensor_t &tensor, float_t value, size_t size) {
  for (auto &t : tensor) {
    t.resize(size, value);
  }
}

enum class net_phase { train, test };

void printTensor(tensor_t& tensor) {
    
    for (int i = 0; i<tensor.size(); i++) {
        std::cout << std::endl;
        for (int j = 0; j<tensor[i].size(); j++) {
            std::cout << " " << tensor[i][j];
        }
    }
    std::cout << std::endl << std::endl;
}

void printMatrix(matrix_t& matrix) {
    
    for (int i = 0; i<matrix.rowCount(); i++) {
        std::cout << std::endl;
        for (int j = 0; j<matrix.colCount(); j++) {
            std::cout << " " << matrix[i * matrix.colCount() + j];
        }
    }
    std::cout << std::endl << std::endl;
}

}