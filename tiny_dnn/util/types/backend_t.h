#pragma once

#include "tiny_dnn/util/error/nn_error.h"

namespace tiny_dnn {
namespace core {

enum class backend_t { internal, nnpack, libdnn, avx, opencl, cblas, intel_mkl };

inline std::ostream &operator<<(std::ostream &os, backend_t type) {
    switch (type) {
        case backend_t::internal: os << "Internal"; break;
        case backend_t::nnpack: os << "NNPACK"; break;
        case backend_t::libdnn: os << "LibDNN"; break;
        case backend_t::avx: os << "AVX"; break;
        case backend_t::opencl: os << "OpenCL"; break;
        case backend_t::cblas: os << "CBLAS"; break;
        case backend_t::intel_mkl: os << "Intel MKL"; break;
        default: throw nn_error("Not supported ostream enum."); break;
    }
    return os;
}

inline backend_t default_engine() {
    #ifdef CNN_USE_AVX
      #if defined(__AVX__) || defined(__AVX2__)
        return backend_t::avx;
      #else
        #error "your compiler does not support AVX"
      #endif
    #else
      return backend_t::internal;
    #endif
}

}
}