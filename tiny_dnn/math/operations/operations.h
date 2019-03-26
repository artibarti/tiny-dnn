#pragma once

#include <map>
#include <algorithm>
#include <vector>

namespace tiny_dnn {
  namespace math {

    enum class Operation {add, subtract, multiply, divide, fill};

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

	bool isSupportedBackend(Operation o, core::backend_t backend) {

        if (o == Operation::add)
            return std::find(supportedBackendsForAdd.begin(), supportedBackendsForAdd.end(),
                backend) != supportedBackendsForAdd.end();
        
        else if (o == Operation::subtract)
            return std::find(supportedBackendsForSubtract.begin(), supportedBackendsForSubtract.end(),
                backend) != supportedBackendsForSubtract.end();
        
        else if (o == Operation::multiply)
            return std::find(supportedBackendsForMultiply.begin(), supportedBackendsForMultiply.end(),
                backend) != supportedBackendsForMultiply.end();

        else if (o == Operation::fill)
            return std::find(supportedBackendsForFill.begin(), supportedBackendsForFill.end(),
                backend) != supportedBackendsForFill.end();
        else 
            return false;
	}

  }
}