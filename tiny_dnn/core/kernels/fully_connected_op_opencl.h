/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#ifdef USE_OPENCL
  #include "tiny_dnn/opencl_util/opencl_util.h"
#endif

#include "tiny_dnn/core/params/fully_params.h"
#include <iostream>

namespace tiny_dnn {
namespace kernels {

inline void fully_connected_op_opencl(const tensor_t &in_data, const vec_t &W, 
    const vec_t &bias, tensor_t &out_data, const core::fully_params &params) {

    #ifdef USE_OPENCL

        KernelGroup kernelGroup = KernelManager::getInstance()
            .getKernelGroup("matrix_operations.cl");
        
        Kernel kernel = kernelGroup.getKernel("matrixMul");

        int inDataElementCount = in_data.size() * params.in_size_;
        int outDataElementCount = out_data.size() * params.out_size_;

        Buffer<float_t> d_in = Buffer<float_t>(kernel.getContext(),
            kernel.getQueue(), inDataElementCount);
        Buffer<float_t> d_out = Buffer<float_t>(kernel.getContext(),
            kernel.getQueue(), outDataElementCount);
        Buffer<float_t> d_W = Buffer<float_t>(kernel.getContext(),
            kernel.getQueue(), W.size());

        for (unsigned int i = 0; i<in_data.size(); i++)
            d_in.write(in_data[i].data(), params.in_size_, i * params.in_size_);        
        d_W.write(W.data(), W.size());

        // thread configuration
        std::vector<size_t> local = {BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z};    
        std::vector<size_t> global = {BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z};

        kernel.setArgument<int>(0, BLOCK_SIZE_X);
        kernel.setArgument<int>(1, BLOCK_SIZE_Y);
        kernel.setArgument<int>(2, BLOCK_SIZE_Z);
        kernel.setArgument<float_t>(3, d_in);
        kernel.setArgument<float_t>(4, d_W);
        kernel.setArgument<float_t>(5, d_out);
        kernel.setArgument<int>(6, in_data.size());
        kernel.setArgument<int>(7, params.in_size_);
        kernel.setArgument<int>(8, params.in_size_);
        kernel.setArgument<int>(9, params.out_size_);
        
        kernel.launch(global, local);

        // read result from device
        for (unsigned int i = 0; i<out_data.size(); i++)
            d_out.read(out_data[i].data(), params.out_size_, i * params.out_size_);

    #else
        throw tiny_dnn::nn_error("tiny-dnn was not built with OpenCL support");
    #endif
}

inline void fully_connected_op_opencl(const tensor_t &prev_out, const vec_t &W,
    tensor_t &dW, tensor_t &db, tensor_t &curr_delta, tensor_t &prev_delta,
    const core::fully_params &params, const bool layer_parallelize) {
    
    for (size_t sample = 0; sample < prev_out.size(); sample++) {
        for (size_t c = 0; c < params.in_size_; c++) {
            // propagate delta to previous layer
            // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
            prev_delta[sample][c] += vectorize::dot(
                &curr_delta[sample][0], &W[c * params.out_size_], params.out_size_);
        }

        for_(layer_parallelize, 0, params.out_size_, [&](const blocked_range &r) {
            // accumulate weight-step using delta
            // dW[c * out_size + i] += current_delta[i] * prev_out[c]
            for (size_t c = 0; c < params.in_size_; c++) {
                vectorize::muladd(&curr_delta[sample][r.begin()], prev_out[sample][c],
                    r.end() - r.begin(), &dW[sample][c * params.out_size_ + r.begin()]);
            }

            if (params.has_bias_) {
                // vec_t& db = *in_grad[2];
                for (size_t i = r.begin(); i < r.end(); i++) {
                    db[sample][i] += curr_delta[sample][i];
                }
            }
        });
    } 
}

}  // namespace kernels
}  // namespace tiny_dnn
