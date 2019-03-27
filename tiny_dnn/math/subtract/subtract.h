#pragma once

#include <stdexcept>

#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/containers/containers.h"
#include "tiny_dnn/util/nn_error.h"

#include "tiny_dnn/math/subtract/subtract_internal.h"

namespace tiny_dnn {
  namespace math {

	template<typename T = float_t, core::backend_t backend = core::backend_t::internal>
	void subtract(const Matrix<T>& left, const Matrix<T>& right, Matrix<T>& result, 
		bool resizeResultIfNeeded = false) {

			if (!isSupportedBackend(Operation::subtract, backend)) {
				throw nn_error("Backend type is not supported for this operation");
			}

			if (!left.hasSameDimensionWith(right)) {
				throw std::invalid_argument("Matrices are not compatible for this operation");
			}

			if (!left.hasSameDimensionWith(result) && !resizeResultIfNeeded) {
				throw std::invalid_argument("Matrices are not compatible for this operation");
			} else if (!left.hasSameDimensionWith(result) && resizeResultIfNeeded) {
				result.resize(left.rowCount(), left.colCount());
			}

			if (backend == core::backend_t::internal) {
				subtract_internal<T>(left, right, result);
			}
	}

  }
}