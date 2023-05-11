#pragma once

#include "Utils/kwTypes.h"

#include "cuda_runtime.h"


namespace kw
{


enum class kDevice { CPU, GPU };
constexpr auto CPU = kDevice::CPU;
constexpr auto GPU = kDevice::GPU;

Error
    fromCudaError(const cudaError_t& cudaError);

}
