#pragma once

#include <vector>
#include "tiny_dnn/util/types/aligned_allocator.h"

namespace tiny_dnn {

#ifdef CNN_USE_DOUBLE
  typedef double float_t;
#else
  typedef float float_t;
#endif

// output label(class-index) for classification must be equal
// to size_t, because size of last layer is equal to number of classes
typedef size_t label_t;

typedef size_t layer_size_t;  // for backward compatibility
typedef std::vector<float_t, aligned_allocator<float_t, 64>> vec_t;
typedef std::vector<vec_t> tensor_t;


}