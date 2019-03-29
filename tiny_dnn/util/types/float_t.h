#pragma once

namespace tiny_dnn {

  #ifdef CNN_USE_DOUBLE
   using float_t = double;
  #else
   using float_t = float;
  #endif

}