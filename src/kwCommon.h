#pragma once

#include <string>

namespace kw
{

using Error = std::string;

enum class kDevice { CPU, GPU };
constexpr auto CPU = kDevice::CPU;
constexpr auto GPU = kDevice::GPU;

Error
    fromCudaError(const cudaError_t& cudaError);

}
