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
  
}