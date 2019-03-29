#pragma once

#include <stdexcept>

#include "tiny_dnn/util/error/nn_error.h"
#include "tiny_dnn/util/types/types.h"
#include "tiny_dnn/math/vector_operations/operations/operations.h"
#include "tiny_dnn/math/vector_operations/multiply/multiply_internal.h"

namespace tiny_dnn {
  namespace math {

	template<core::backend_t backend = core::backend_t::internal>
	void multiply(const tensor_t& left, const tensor_t& right, tensor_t& result, 
		bool resizeResultIfNeeded = false) {

			if (!isSupportedBackend(Operation::multiply, backend)) {
				throw nn_error("Backend type is not supported for this operation");
			}

			shape2d left_shape = getDimension(left);
			shape2d right_shape = getDimension(right);
			shape2d result_shape = getDimension(result);

			if (left_shape == 0 || right_shape == 0) {
				throw std::invalid_argument("Operation is not supported for empty tensors");
			}
			if (left_shape.y != right_shape.x) {
				throw std::invalid_argument("Tensors are not compatible for this operation");
			}
			if (result_shape == 0 && resizeResultIfNeeded) {
				// TODO resize
			}
			if (result_shape.x != left_shape.x && result_shape.y != right_shape.y
				&& !resizeResultIfNeeded) {
				throw std::invalid_argument("Tensors are not compatible for this operation");					
			}      
			
			if (backend == core::backend_t::internal) {
				multiply_internal(left, right, result);
			}
	}
  
	}
}