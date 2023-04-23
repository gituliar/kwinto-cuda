#include "cuda_runtime.h"

#include "kwCommon.h"


kw::Error
kw::fromCudaError(const cudaError_t& cudaError)
{
    Error error;

    error += cudaGetErrorName(cudaError);
    error += ": ";
    error += cudaGetErrorString(cudaError);

    return error;
}
