#pragma once

namespace tiny_dnn {

// 0x0001XXX : in/out data
// 0x0001000 : input/output data, fed by other layer or input channel
// 0x0002XXX : trainable parameters, updated for each back propagation
// 0x0010000 : layer-specific storage

enum class vector_type : int32_t {
  data = 0x0001000,

  weight = 0x0002000,
  bias   = 0x0002001,

  label = 0x0004000,
  aux   = 0x0010000
};

}