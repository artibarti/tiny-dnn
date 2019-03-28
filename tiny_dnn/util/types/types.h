#pragma once

#include <vector>
#include <limits>
#include <string>

#include "tiny_dnn/util/types/aligned_allocator.h"
#include "tiny_dnn/containers/containers.h"
#include "tiny_dnn/util/error/nn_error.h"

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

template <typename T>
struct index3d {
  
  index3d(T width, T height, T depth) {
    reshape(width, height, depth);
  }

  index3d() : width_(0), height_(0), depth_(0) {}

  void reshape(T width, T height, T depth) {
    width_  = width;
    height_ = height;
    depth_  = depth;

    if ((int64_t)width * height * depth > std::numeric_limits<T>::max()) {
      throw nn_error("error while constructing layer: layer size too large for tiny-dnn");
      // \nWidth x Height x Channels = " << width << ">= max size of "
      //    "[%s](=%d)", width, height, depth, typeid(T).name(), std::numeric_limits<T>::max());
    }
  }

  T get_index(T x, T y, T channel) const {
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
    assert(channel >= 0 && channel < depth_);
    return (height_ * channel + y) * width_ + x;
  }

  T area() const {
    return width_ * height_;
  }

  T size() const {
    return width_ * height_ * depth_;
  }

  T width_;
  T height_;
  T depth_;
};

typedef index3d<size_t> shape3d;

template <typename T>
bool operator==(const index3d<T> &lhs, const index3d<T> &rhs) {
  return (lhs.width_ == rhs.width_) && (lhs.height_ == rhs.height_)
    && (lhs.depth_ == rhs.depth_);
}

template <typename T>
bool operator!=(const index3d<T> &lhs, const index3d<T> &rhs) {
  return !(lhs == rhs);
}

template <typename Stream, typename T>
Stream &operator<<(Stream &s, const index3d<T> &d) {
  s << d.width_ << "x" << d.height_ << "x" << d.depth_;
  return s;
}

template <typename T>
std::ostream &operator<<(std::ostream &s, const index3d<T> &d) {
  s << d.width_ << "x" << d.height_ << "x" << d.depth_;
  return s;
}

template <typename Stream, typename T>
Stream &operator<<(Stream &s, const std::vector<index3d<T>> &d) {
  s << "[";
  for (size_t i = 0; i < d.size(); i++) {
    if (i) s << ",";
    s << "[" << d[i] << "]";
  }
  s << "]";
  return s;
}

}