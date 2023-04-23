#pragma once

#include <string>

namespace kw
{

using Error = std::string;

Error
    fromCudaError(const cudaError_t& cudaError);

}
