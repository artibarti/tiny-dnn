#pragma once

namespace tiny_dnn {

#ifdef CNN_USE_DOUBLE
 typedef double float_t;
#else
 typedef float float_t;
#endif

using vec_t = std::vector<float_t, aligned_allocator<float_t, 64>>;
using tensor_t = std::vector<vec_t>;

struct shape2d
{
    unsigned x, y;

    shape2d() : x(0), y(0) {}
    shape2d(unsigned _x, unsigned _y) : x(_x), y(_y) {}
            
    bool operator==(const shape2d& other) {
        return (x == other.x && y == other.y);
    }    
    bool operator!=(const shape2d& other) {
        return !(x == other.x && y == other.y);
    }
    bool operator==(unsigned size) {
        return (x == size && y == size);
    }    

    unsigned operator[] (unsigned index) {
        if (index == 0) {
            return x;
        } else if (index == 1) {
            return y;
        } else {
            throw std::out_of_range("Index out of bounds");
        }
    }
};

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