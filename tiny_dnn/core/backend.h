/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

#include "tiny_dnn/util/types/types.h"
#include "tiny_dnn/core/params/conv_params.h"
#include "tiny_dnn/core/params/deconv_params.h"
#include "tiny_dnn/core/params/fully_params.h"
#include "tiny_dnn/core/params/global_avepool_params.h"
#include "tiny_dnn/core/params/maxpool_params.h"
#include "tiny_dnn/network/layers/layer.h"
#include "tiny_dnn/network/node.h"

#ifdef CNN_USE_NNPACK
#include <nnpack.h>
#endif

namespace tiny_dnn {
namespace core {

#ifdef CNN_USE_NNPACK
// Singleton to keep a global state whether NNPACK is initialized.
// Before using the API an initialization is required. For this reason
// we need to get an instance of the object in order to avoid a throw error.
//
// Usage:
//     NNPackInitializer::getInstance().initialize();
//
class NNPackInitializer {
 public:
  // We create a static instance of the object in case
  // that it wasn't created before and we return it.
  static NNPackInitializer &getInstance() {
    static NNPackInitializer instance;
    return instance;
  }

  // Tries to initialize NNPACK.
  // Calls an internal method to initialize in case that it's not,
  // otherwise it returns a void.
  // Throws an error if we do not succed with initialization.
  void initialize() {
    if (initialized_) return;  // alredy initialized, do nothig.

    // calls internal method to initialize
    nnp_status init_status = nnp_initialize();
    if (init_status != nnp_status_success) {
      throw nn_error("Cannot initialize NNPACK.");
    }

    // succeded with initialization. We set the global
    // state to avoid exception errors in addition to
    // reuse code.
    initialized_ = true;
  }

 private:
  /** Flag to store whether NNPACK is initialized */
  bool initialized_ = false;
};

// TODO(you): create an interface to let users choose the algorithm
inline nnp_convolution_algorithm nnp_algorithm() {
  return nnp_convolution_algorithm_auto;
}

// TODO(you): create an interface to let users choose the transform strategy
inline nnp_convolution_transform_strategy nnp_kts() {
  // some algorithm accept tuple based only
  return nnp_convolution_transform_strategy_tuple_based;
}
#endif

class backend {

  public:
    explicit backend() {}

  // core math functions

  virtual void conv2d_q(const std::vector<tensor_t *> &in_data,
                        std::vector<tensor_t *> &out_data) = 0;

  virtual void conv2d_eq(const std::vector<tensor_t *> &in_data,
                         std::vector<tensor_t *> &out_data) = 0;

  virtual void conv2d_q(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) = 0;

  virtual void deconv2d(const std::vector<tensor_t *> &in_data,
                        std::vector<tensor_t *> &out_data) = 0;

  virtual void deconv2d_q(const std::vector<tensor_t *> &in_data,
                          std::vector<tensor_t *> &out_data) = 0;

  virtual void deconv2d_eq(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) = 0;

  virtual void deconv2d(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) = 0;

  virtual void deconv2d_q(const std::vector<tensor_t *> &in_data,
                          const std::vector<tensor_t *> &out_data,
                          std::vector<tensor_t *> &out_grad,
                          std::vector<tensor_t *> &in_grad) = 0;

  virtual void fully_q(const std::vector<tensor_t *> &in_data,
                       std::vector<tensor_t *> &out_data) = 0;

  virtual void fully_eq(const std::vector<tensor_t *> &in_data,
                        std::vector<tensor_t *> &out_data) = 0;

  virtual void fully_q(const std::vector<tensor_t *> &in_data,
                       const std::vector<tensor_t *> &out_data,
                       std::vector<tensor_t *> &out_grad,
                       std::vector<tensor_t *> &in_grad) = 0;

  void set_layer(layer *layer) { layer_ = layer; }

  virtual backend_t type() const = 0;

 protected:
  layer *layer_;
};

}  // namespace core
}  // namespace tiny_dnn
