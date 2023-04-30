#pragma once

#include <string>

#include "cuda_runtime.h"


namespace kw
{

using Error = std::string;

enum class kDevice { CPU, GPU };
constexpr auto CPU = kDevice::CPU;
constexpr auto GPU = kDevice::GPU;

Error
    fromCudaError(const cudaError_t& cudaError);

}
