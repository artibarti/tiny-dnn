#pragma once

#include <stdexcept>

#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/util/error/nn_error.h"
#include "tiny_dnn/util/types/types.h"
#include "tiny_dnn/math/vector_operations/operations/operations.h"
#include "tiny_dnn/math/vector_operations/subtract/subtract_internal.h"

namespace tiny_dnn {
  namespace math {

	template<core::backend_t backend = core::backend_t::internal>
	void subtract(const tensor_t& left, const tensor_t& right, tensor_t& result, 
		bool resizeResultIfNeeded = false) {

			if (!isSupportedBackend(Operation::subtract, backend)) {
				throw nn_error("Backend type is not supported for this operation");
			}
			if (!haveSameDimensions(left, right)) {
				throw std::invalid_argument("Tensors are not compatible for this operation");
			}
			if (!haveSameDimensions(left, result)) {
				if (!resizeResultIfNeeded) {
					throw std::invalid_argument("Tensors are not compatible for this operation");
				} else {
					// TODO resize
				}
			}

			if (backend == core::backend_t::internal) {
				subtract_internal(left, right, result);
			}
	}

  }
}