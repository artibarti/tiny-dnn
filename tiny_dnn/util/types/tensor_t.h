#pragma once

#include <iostream>

#include "tiny_dnn/util/types/aligned_allocator.h"
#include "tiny_dnn/util/types/float_t.h"
#include "tiny_dnn/util/types/index2d.h"

namespace tiny_dnn {

using vec_t = std::vector<float_t, aligned_allocator<float_t, 64>>;
using tensor_t = std::vector<vec_t>;

shape2d getDimension(const tensor_t& tensor) {
    if (tensor.size() != 0) {
        unsigned cols = tensor[0].size();
        for (unsigned i = 1; i < tensor.size(); i++) {
            if (tensor[i].size() != cols) {
                throw std::runtime_error("Tensor is not a valid matrix");
            }
        }
        return shape2d(tensor.size(), cols);
    }
    return shape2d(0,0);
}

bool haveSameDimensions(const tensor_t& t1, const tensor_t& t2) {
    return getDimension(t1) == getDimension(t2);
}

}