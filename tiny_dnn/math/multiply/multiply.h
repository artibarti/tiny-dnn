#pragma once

#include <stdexcept>

#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/containers/containers.h"
#include "tiny_dnn/util/nn_error.h"

#include "tiny_dnn/math/multiply/multiply_internal.h"

namespace tiny_dnn {
  namespace math {

	template<typename T, core::backend_t backend = core::backend_t::internal>
	void multiply(const Matrix<T>& left, const Matrix<T>& right, Matrix<T>& result, 
		bool resizeResultIfNeeded = false) {

			if (!isSupportedBackend(Operation::multiply, backend)) {
				throw nn_error("Backend type is not supported for this operation");
			}

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

			if (backend == core::backend_t::internal) {
				multiply_internal<T>(left, right, result);
			}
	}
  
	}
}