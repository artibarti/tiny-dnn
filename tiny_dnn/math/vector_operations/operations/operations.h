#pragma once

#include <map>
#include <algorithm>
#include <vector>

#include "tiny_dnn/util/types/types.h"

namespace tiny_dnn {
  namespace math {

    enum class Operation { add, subtract, multiply, fill };

    std::vector<core::backend_t> supportedBackendsForAdd = {
        core::backend_t::internal
    };

    std::vector<core::backend_t> supportedBackendsForSubtract = {
        core::backend_t::internal
    };

    std::vector<core::backend_t> supportedBackendsForMultiply = {
        core::backend_t::internal
    };

    std::vector<core::backend_t> supportedBackendsForFill = {
        core::backend_t::internal
    };

	bool isSupportedBackend(Operation operation, core::backend_t backend) {

        if (operation == Operation::add)
            return std::find(supportedBackendsForAdd.begin(), supportedBackendsForAdd.end(),
                backend) != supportedBackendsForAdd.end();
        
        else if (operation == Operation::subtract)
            return std::find(supportedBackendsForSubtract.begin(), supportedBackendsForSubtract.end(),
                backend) != supportedBackendsForSubtract.end();
        
        else if (operation == Operation::multiply)
            return std::find(supportedBackendsForMultiply.begin(), supportedBackendsForMultiply.end(),
                backend) != supportedBackendsForMultiply.end();

        else if (operation == Operation::fill)
            return std::find(supportedBackendsForFill.begin(), supportedBackendsForFill.end(),
                backend) != supportedBackendsForFill.end();
        else 
            return false;
	}

  }
}