#pragma once

#include <vector>
#include <limits>
#include <string>

#include "tiny_dnn/util/types/aligned_allocator.h"
#include "tiny_dnn/util/types/tensor_t.h"
#include "tiny_dnn/util/types/index3d.h"

namespace tiny_dnn {

using label_t = size_t;

enum class net_phase {
  train, test
};

enum class padding {
  valid,  // use valid pixels of input
  same    // add zero-padding around input so as to keep image size
};

}