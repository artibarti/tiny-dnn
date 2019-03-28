#pragma once

namespace tiny_dnn {

#ifdef CNN_USE_DOUBLE
 typedef double float_t;
#else
 typedef float float_t;
#endif

using vec_t = std::vector<float_t, aligned_allocator<float_t, 64>>;
using tensor_t = std::vector<vec_t>;

bool hasSameDimensions(vec_t& v1, vec_t& v2) {
    return v1.size() == v2.size();
}

bool hasSameDimensions(tensor_t& t1, tensor_t& t2) {
    if (t1.size() == t2.size()) {
        if (t1.size() != 0) {
            if (t1[0].size() == t2[0].size()) {
                return true;
            }
        }
    }
    return false;
}

bool isMultiplicable(tensor_t& t1, tensor_t& t2) {
    if (t1.size() != 0 && t2.size() != 0) {
        if (t1[0].size() == t2.size()) {
            return true;
        }
    }
    return false;
}

}