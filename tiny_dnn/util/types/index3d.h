#pragma once

#include <cassert>

namespace tiny_dnn {

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
      throw std::runtime_error("error while constructing layer: layer size too large for tiny-dnn");
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

using shape3d = index3d<size_t>;

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