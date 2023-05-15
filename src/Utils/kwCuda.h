#pragma once

#include "Utils/kwTypes.h"

#include "cuda_runtime.h"


namespace kw
{

Error
    fromCudaError(const cudaError_t& cudaError);

}
