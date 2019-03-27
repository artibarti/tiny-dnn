#pragma once

#include <stdexcept>

#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/containers/containers.h"
#include "tiny_dnn/util/nn_error.h"

#include "tiny_dnn/math/fill/fill_internal.h"

namespace tiny_dnn {
  namespace math {

	template<typename T = float_t, core::backend_t backend = core::backend_t::internal>
	void fill(Matrix<T>& mat, const T& value = 0) {

			if (!isSupportedBackend(Operation::fill, backend)) {
				throw nn_error("Backend type is not supported for this operation");
			}

			if (backend == core::backend_t::internal) {
				fill_internal<T>(mat, value);
			}
	}
  
	}
}