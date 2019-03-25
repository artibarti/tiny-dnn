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

        // TODO avoid double copy
        vec_t in, out;
        for (unsigned int i = 0; i<in_data.size(); i++)
            for (unsigned int j = 0; j<in_data[i].size(); j++)
                in.push_back(in_data[i][j]);
        for (unsigned int i = 0; i<out_data.size(); i++)
            for (unsigned int j = 0; j<out_data[i].size(); j++)
                out.push_back(out_data[i][j]);

        // TODO vectors should be passed directly to setArgument
        auto d_in = CLCudaAPI::Buffer<float>(*(kernel.getContext()),
            *(kernel.getQueue()), in.begin(), in.end());
        auto d_W = CLCudaAPI::Buffer<float>(*(kernel.getContext()),
            *(kernel.getQueue()), W.begin(), W.end());
        auto d_out = CLCudaAPI::Buffer<float>(*(kernel.getContext()),
            *(kernel.getQueue()), out.begin(), out.end());

        int blockSize = KernelManager::BLOCK_SIZE;

        // thread configuration
        std::vector<size_t> local = {blockSize, blockSize,1};    
        std::vector<size_t> global = {blockSize, blockSize, 1};

        kernel.setArgument<int>(0, blockSize);
        kernel.setArgument<float>(1, &d_in);
        kernel.setArgument<float>(2, &d_W);
        kernel.setArgument<float>(3, &d_out);
        kernel.setArgument<int>(4, in_data.size());
        kernel.setArgument<int>(5, params.in_size_);
        kernel.setArgument<int>(6, params.in_size_);
        kernel.setArgument<int>(7, params.out_size_);
        
        kernel.launch(global, local);

        // read the result back into out_data 
        std::vector<float> result(out.size());
        d_out.Read(*(kernel.getQueue()), out.size(), result);

        for (unsigned int i = 0; i < in_data.size(); i++) {
            vec_t out_row;
            for (unsigned int j = 0; j < params.out_size_; j++) {
                out_row.push_back(result[i * params.out_size_ + j]);
            }
            out_data[i] = out_row;
        }    
    
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
