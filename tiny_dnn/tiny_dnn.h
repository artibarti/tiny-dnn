/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/util/util.h"
#include "tiny_dnn/math/math.h"

#ifdef USE_OPENCL
#include "tiny_dnn/opencl_util/opencl_util.h"
#endif

#include "tiny_dnn/optimizers/optimizer.h"
#include "tiny_dnn/network/network.h"

#include "tiny_dnn/network/activations/asinh_layer.h"
#include "tiny_dnn/network/activations/elu_layer.h"
#include "tiny_dnn/network/activations/leaky_relu_layer.h"
#include "tiny_dnn/network/activations/relu_layer.h"
#include "tiny_dnn/network/activations/selu_layer.h"
#include "tiny_dnn/network/activations/sigmoid_layer.h"
#include "tiny_dnn/network/activations/softmax_layer.h"
#include "tiny_dnn/network/activations/softplus_layer.h"
#include "tiny_dnn/network/activations/softsign_layer.h"
#include "tiny_dnn/network/activations/tanh_layer.h"
#include "tiny_dnn/network/activations/tanh_p1m2_layer.h"
#include "tiny_dnn/network/layers/arithmetic_layer.h"
#include "tiny_dnn/network/layers/average_pooling_layer.h"
#include "tiny_dnn/network/layers/average_unpooling_layer.h"
#include "tiny_dnn/network/layers/batch_normalization_layer.h"
#include "tiny_dnn/network/layers/cell.h"
#include "tiny_dnn/network/layers/cells.h"
#include "tiny_dnn/network/layers/concat_layer.h"
#include "tiny_dnn/network/layers/convolutional_layer.h"
#include "tiny_dnn/network/layers/deconvolutional_layer.h"
#include "tiny_dnn/network/layers/dropout_layer.h"
#include "tiny_dnn/network/layers/fully_connected_layer.h"
#include "tiny_dnn/network/layers/global_average_pooling_layer.h"
#include "tiny_dnn/network/layers/input_layer.h"
#include "tiny_dnn/network/layers/l2_normalization_layer.h"
#include "tiny_dnn/network/layers/lrn_layer.h"
#include "tiny_dnn/network/layers/lrn_layer.h"
#include "tiny_dnn/network/layers/max_pooling_layer.h"
#include "tiny_dnn/network/layers/max_unpooling_layer.h"
#include "tiny_dnn/network/layers/power_layer.h"
#include "tiny_dnn/network/layers/quantized_convolutional_layer.h"
#include "tiny_dnn/network/layers/quantized_deconvolutional_layer.h"
#include "tiny_dnn/network/layers/recurrent_layer.h"
#include "tiny_dnn/network/layers/slice_layer.h"
#include "tiny_dnn/network/layers/zero_pad_layer.h"

#ifdef CNN_USE_GEMMLOWP
  #include "tiny_dnn/network/layers/quantized_fully_connected_layer.h"
#endif

#include "tiny_dnn/io/cifar10_parser.h"
#include "tiny_dnn/io/display.h"
#include "tiny_dnn/io/layer_factory.h"
#include "tiny_dnn/io/mnist_parser.h"

#ifndef CNN_NO_SERIALIZATION
  #include "tiny_dnn/serialization/serialization.h"
  CEREAL_REGISTER_TYPE(tiny_dnn::elu_layer)
  CEREAL_REGISTER_TYPE(tiny_dnn::leaky_relu_layer)
  CEREAL_REGISTER_TYPE(tiny_dnn::relu_layer)
  CEREAL_REGISTER_TYPE(tiny_dnn::sigmoid_layer)
  CEREAL_REGISTER_TYPE(tiny_dnn::softmax_layer)
  CEREAL_REGISTER_TYPE(tiny_dnn::softplus_layer)
  CEREAL_REGISTER_TYPE(tiny_dnn::softsign_layer)
  CEREAL_REGISTER_TYPE(tiny_dnn::tanh_layer)
  CEREAL_REGISTER_TYPE(tiny_dnn::tanh_p1m2_layer)
#endif

// shortcut version of layer names
namespace tiny_dnn {
 namespace layers {
  using conv = tiny_dnn::convolutional_layer;
  using q_conv = tiny_dnn::quantized_convolutional_layer;
  using max_pool = tiny_dnn::max_pooling_layer;
  using ave_pool = tiny_dnn::average_pooling_layer;
  using fc = tiny_dnn::fully_connected_layer;
  using dense = tiny_dnn::fully_connected_layer;
  using zero_pad = tiny_dnn::zero_pad_layer;
  #ifdef CNN_USE_GEMMLOWP
    using q_fc = tiny_dnn::quantized_fully_connected_layer;
  #endif
  using add = tiny_dnn::elementwise_add_layer;
  using dropout = tiny_dnn::dropout_layer;
  using input = tiny_dnn::input_layer;
  using linear = linear_layer;
  using lrn = tiny_dnn::lrn_layer;
  using concat = tiny_dnn::concat_layer;
  using deconv = tiny_dnn::deconvolutional_layer;
  using max_unpool = tiny_dnn::max_unpooling_layer;
  using ave_unpool = tiny_dnn::average_unpooling_layer;
 }  // namespace layers
 namespace activation {
  using sigmoid = tiny_dnn::sigmoid_layer;
  using asinh = tiny_dnn::asinh_layer;
  using tanh = tiny_dnn::tanh_layer;
  using relu = tiny_dnn::relu_layer;
  using rectified_linear = tiny_dnn::relu_layer;
  using softmax = tiny_dnn::softmax_layer;
  using leaky_relu = tiny_dnn::leaky_relu_layer;
  using elu = tiny_dnn::elu_layer;
  using selu = tiny_dnn::selu_layer;
  using tanh_p1m2 = tiny_dnn::tanh_p1m2_layer;
  using softplus = tiny_dnn::softplus_layer;
  using softsign = tiny_dnn::softsign_layer;
 }  // namespace activation

 #include "tiny_dnn/models/alexnet.h"

 using batch_norm = tiny_dnn::batch_normalization_layer;
 using l2_norm = tiny_dnn::l2_normalization_layer;
 using slice = tiny_dnn::slice_layer;
 using power = tiny_dnn::power_layer;
}  // namespace tiny_dnn

#ifdef CNN_USE_CAFFE_CONVERTER
  // experimental / require google protobuf
  #include "tiny_dnn/io/caffe/layer_factory.h"
#endif
