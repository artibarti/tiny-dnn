/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "tiny_dnn/util/config/config.h"
#include "tiny_dnn/util/types/types.h"
#include "tiny_dnn/util/error/nn_error.h"
#include "tiny_dnn/util/functions/functions.h"

#ifndef CNN_NO_SERIALIZATION
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#endif

namespace tiny_dnn {

template <typename T>
T *reverse_endian(T *p) {
  std::reverse(reinterpret_cast<char *>(p), reinterpret_cast<char *>(p) + sizeof(T));
  return p;
}

inline bool is_little_endian() {
  int x = 1;
  return *reinterpret_cast<char *>(&x) != 0;
}

template <typename T>
size_t max_index(const T &vec) {
  auto begin_iterator = std::begin(vec);
  return std::max_element(begin_iterator, std::end(vec)) - begin_iterator;
}

template <typename T, typename U>
U rescale(T x, T src_min, T src_max, U dst_min, U dst_max) {
  U value = static_cast<U>(
    ((x - src_min) * (dst_max - dst_min)) / (src_max - src_min) + dst_min);
  return std::min(dst_max, std::max(value, dst_min));
}

inline void nop() {}

template <typename T>
inline T sqr(T value) {
  return value * value;
}

inline bool isfinite(float_t x) {
  return x == x;
}

template <typename Container>
inline bool has_infinite(const Container &c) {
  for (auto v : c)
    if (!isfinite(v))
      return true;
  return false;
}

template <typename Container>
size_t max_size(const Container &c) {
  typedef typename Container::value_type value_t;
  const auto max_size = std::max_element(c.begin(), c.end(),
    [](const value_t &left, const value_t &right) {
      return left.size() < right.size();
    })->size();
  assert(max_size <= std::numeric_limits<size_t>::max());
  return max_size;
}

inline std::string format_str(const char *fmt, ...) {
  
  static char buf[2048];

  #ifdef _MSC_VER
    #pragma warning(disable : 4996)
  #endif
  
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);

  #ifdef _MSC_VER
    #pragma warning(default : 4996)
  #endif
  
  return std::string(buf);
}

// equivalent to std::to_string, which android NDK doesn't support
template <typename T>
std::string to_string(T value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

template <typename T, typename Pred, typename Sum>
size_t sumif(const std::vector<T> &vec, Pred p, Sum s) {
  size_t sum = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    if (p(i)) sum += s(vec[i]);
  }
  return sum;
}

template <typename T, typename Pred>
std::vector<T> filter(const std::vector<T> &vec, Pred p) {
  std::vector<T> res;
  for (size_t i = 0; i < vec.size(); i++) {
    if (p(i)) res.push_back(vec[i]);
  }
  return res;
}

template <typename Result, typename T, typename Pred>
std::vector<Result> map_(const std::vector<T> &vec, Pred p) {
  std::vector<Result> res(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    res[i] = p(vec[i]);
  }
  return res;
}

enum class vector_type : int32_t {
  // 0x0001XXX : in/out data
  data = 0x0001000,  // input/output data, fed by other layer or input channel

  // 0x0002XXX : trainable parameters, updated for each back propagation
  weight = 0x0002000,
  bias   = 0x0002001,

  label = 0x0004000,
  aux   = 0x0010000  // layer-specific storage
};

inline std::string to_string(vector_type vtype) {
  switch (vtype) {
    case tiny_dnn::vector_type::data: return "data";
    case tiny_dnn::vector_type::weight: return "weight";
    case tiny_dnn::vector_type::bias: return "bias";
    case tiny_dnn::vector_type::label: return "label";
    case tiny_dnn::vector_type::aux: return "aux";
    default: return "unknown";
  }
}

inline std::ostream &operator<<(std::ostream &os, vector_type vtype) {
  os << to_string(vtype);
  return os;
}

inline vector_type operator&(vector_type lhs, vector_type rhs) {
  return (vector_type)(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

inline bool is_trainable_weight(vector_type vtype) {
  return (vtype & vector_type::weight) == vector_type::weight;
}

inline std::vector<vector_type> std_input_order(bool has_bias) {
  if (has_bias) {
    return {vector_type::data, vector_type::weight, vector_type::bias};
  } else {
    return {vector_type::data, vector_type::weight};
  }
}

inline void fill_tensor(tensor_t &tensor, float_t value) {
  for (auto &t : tensor) {
    vectorize::fill(&t[0], t.size(), value);
  }
}

inline void fill_tensor(tensor_t &tensor, float_t value, size_t size) {
  for (auto &t : tensor) {
    t.resize(size, value);
  }
}

inline size_t conv_out_length(size_t in_length, size_t window_size,
  size_t stride, size_t dilation, padding pad_type) {
  
  size_t output_length;
  if (pad_type == padding::same) {
    output_length = in_length;
  } else if (pad_type == padding::valid) {
    output_length = in_length - dilation * window_size + dilation;
  } else {
    throw nn_error("Not recognized pad_type.");
  }
  return (output_length + stride - 1) / stride;
}

inline size_t pool_out_length(size_t in_length, size_t window_size,
  size_t stride, bool ceil_mode, padding pad_type) {
  
  size_t output_length;

  if (pad_type == padding::same) {
    output_length = in_length;
  } else if (pad_type == padding::valid) {
    output_length = in_length - window_size + 1;
  } else {
    throw nn_error("Not recognized pad_type.");
  }

  float tmp = static_cast<float>((output_length + stride - 1)) / stride;
  return static_cast<int>(ceil_mode ? ceil(tmp) : floor(tmp));
}

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class ValType, class T>
using value_type_is = std::enable_if_t<std::is_same<T, typename ValType::value_type>::value>;

template <class ValType>
using value_is_float = value_type_is<ValType, float>;

template <class ValType>
using value_is_double = value_type_is<ValType, double>;

template <template <typename> class checker, typename... Ts>
struct are_all : std::true_type {};

template <template <typename> class checker, typename T0, typename... Ts>
struct are_all<checker, T0, Ts...>
  : std::integral_constant<bool, checker<T0>::value && are_all<checker, Ts...>::value> {};

}