#pragma once

#include <stdexcept>

#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/util/error/nn_error.h"
#include "tiny_dnn/util/types/types.h"
#include "tiny_dnn/math/vector_operations/operations/operations.h"
#include "tiny_dnn/math/vector_operations/add/add_internal.h"

namespace tiny_dnn {
 namespace math {

	template<core::backend_t backend = core::backend_t::internal>
	void add(const tensor_t& left, const tensor_t& right, tensor_t& result, 
		bool resizeResultIfNeeded = false) {
			
			if (!isSupportedBackend(Operation::add, backend)) {
				throw nn_error("Backend type is not supported for this operation");
			}

			/*
			if (!left.hasSameDimensionWith(right)) {
				throw std::invalid_argument("Matrices are not compatible for this operation");
			}

			if (!left.hasSameDimensionWith(result) && !resizeResultIfNeeded) {
				throw std::invalid_argument("Matrices are not compatible for this operation");
			} else if (!left.hasSameDimensionWith(result) && resizeResultIfNeeded) {
				result.resize(left.rowCount(), left.colCount());
			}
			*/

			if (backend == core::backend_t::internal) {
				add_internal(left, right, result);
			}
	}

 }
}