#pragma once

#include <stdexcept>

#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/util/error/nn_error.h"
#include "tiny_dnn/util/types/types.h"
#include "tiny_dnn/math/vector_operations/operations/operations.h"
#include "tiny_dnn/math/vector_operations/fill/fill_internal.h"

namespace tiny_dnn {
  namespace math {

	template<core::backend_t backend = core::backend_t::internal>
	void fill(tensor_t& mat, const float_t& value = 0) {

		if (!isSupportedBackend(Operation::fill, backend)) {
			throw nn_error("Backend type is not supported for this operation");
		}

		if (backend == core::backend_t::internal) {
			fill_internal(mat, value);
		}
	}

	}
}