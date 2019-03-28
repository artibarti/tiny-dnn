#pragma once

#include <vector>
#include "tiny_dnn/util/types/aligned_allocator.h"
#include "tiny_dnn/containers/containers.h"

namespace tiny_dnn {

#ifdef CNN_USE_DOUBLE
  typedef double float_t;
#else
  typedef float float_t;
#endif

// output label(class-index) for classification must be equal
// to size_t, because size of last layer is equal to number of classes
using label_t = size_t;
using vec_t = std::vector<float_t, aligned_allocator<float_t, 64>>;
using tensor_t = std::vector<vec_t>;
using matrix_t = Matrix<float_t>;

enum class net_phase { train, test };

enum class padding {
  valid,  ///< use valid pixels of input
  same    ///< add zero-padding around input so as to keep image size
};

}