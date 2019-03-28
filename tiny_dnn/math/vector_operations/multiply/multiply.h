#pragma once

#include <stdexcept>

#include "tiny_dnn/core/backend.h"
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

			/*
			if (!left.isMultipliableWith(right)) {
				throw std::invalid_argument("Matrices are not compatible for this operation");
			}

      bool resultSizeIsCorrect = result.rowCount() != left.rowCount()
          && result.colCount() != right.colCount();

			if (resultSizeIsCorrect && !resizeResultIfNeeded) {
				throw std::invalid_argument("Matrices are not compatible for this operation");
			} else if (resultSizeIsCorrect && resizeResultIfNeeded) {
				result.resize(left.rowCount(), right.colCount());
			}
			*/
			
			if (backend == core::backend_t::internal) {
				multiply_internal(left, right, result);
			}
	}
  
	}
}