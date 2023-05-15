#include "Utils/kwVector2d.h"

#include "cuda.h"

#include <cstdio>


namespace kw
{

template<typename Real>
__global__
void
adjustEarlyExercise_cuda(
    const size_t n,
    const size_t xDim,
    const Real* pay,
    Real* v)
{
    // pde index
    size_t ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= n)
        return;

    // x-record index
    size_t xi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi >= xDim)
        return;

    const auto i = ni + xi * n;
    if (v[i] < pay[i])
        v[i] = pay[i];
}

// Calc B = [1 - θ dt 𝒜]
template<typename Real>
__global__
void
fillB_cuda(
    const size_t n,
    const size_t xDim,
    const size_t ti,
    const Real theta,

    const Real* t,
    const Real* x,
    const Real* a0,
    const Real* ax,
    const Real* axx,
    const Real* v,

    Real* w,
    Real* bl,
    Real* b,
    Real* bu)
{
    // pde index
    size_t ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= n)
        return;

    // x-record index
    size_t xi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi >= xDim)
        return;

    const auto i = ni + xi * n;
    const auto j = ni + ti * n;

    const Real dt = t[j + n] - t[j];

    if (xi == 0) {
        const Real inv_dx = 1. / (x[i + n] - x[i]);

        bl[i] = 0;
        b[i] = 1 - theta * dt * (a0[j] - inv_dx * ax[j]);
        bu[i] = -theta * dt * (inv_dx * ax[j]);

        w[i] = v[i] + (1 - theta) * dt * (a0[j] * v[i] + (inv_dx * ax[j]) * (v[i + n] - v[i]));
    }
    else if (xi < xDim - 1) {
        const Real inv_dxu = 1. / (x[i + n] - x[i]);
        const Real inv_dxm = 1. / (x[i + n] - x[i - n]);
        const Real inv_dxd = 1. / (x[i] - x[i - n]);

        const Real inv_dx2u = (Real)(2.) * inv_dxu * inv_dxm;
        const Real inv_dx2m = (Real)(2.) * inv_dxd * inv_dxu;
        const Real inv_dx2d = (Real)(2.) * inv_dxd * inv_dxm;

        bl[i] = -theta * dt * (-inv_dxm * ax[j] + inv_dx2d * axx[j]);
        b[i] = 1 - theta * dt * (a0[j] - inv_dx2m * axx[j]);
        bu[i] = -theta * dt * (inv_dxm * ax[j] + inv_dx2u * axx[j]);

        w[i] = v[i] + (1 - theta) * dt * (a0[j] * v[i] + (ax[j] * inv_dxm) * (v[i + n] - v[i - n]) +
            (axx[j]) * (inv_dx2u * v[i + n] - inv_dx2m * v[i] + inv_dx2d * v[i - n]));
    }
    else {
        const Real inv_dx = 1. / (x[i] - x[i - n]);

        bl[i] = -theta * dt * (-inv_dx * ax[j]);
        b[i] = 1 - theta * dt * (a0[j] + inv_dx * ax[j]);
        bu[i] = 0;

        w[i] = v[i] + (1 - theta) * dt * (a0[j] * v[i] + (inv_dx * ax[j]) * (v[i] - v[i - n]));
    }
};


template<typename Real>
Error
adjustEarlyExercise(
    const dim3 block2d,
    const size_t bDim,
    const size_t xDim,
    const Real* pay,
    Real* v)
{
    dim3 grid2d(
        (bDim + block2d.x - 1) / block2d.x,
        (xDim + block2d.y - 1) / block2d.y);

    adjustEarlyExercise_cuda<<<grid2d, block2d>>>(bDim, xDim , pay, v);
    cudaDeviceSynchronize();

    return "";
}

template<typename Real>
Error
fillB(
    const dim3 block2d,
    const size_t bDim,
    const size_t xDim,
    const size_t ti,
    const Real theta,
    const Real* t,
    const Real* x,
    const Real* a0,
    const Real* ax,
    const Real* axx,
    const Real* v,
    Real* w,
    Real* bl,
    Real* b,
    Real* bu)
{
    dim3 grid2d(
        (bDim + block2d.x - 1) / block2d.x,
        (xDim + block2d.y - 1) / block2d.y);

    fillB_cuda<<<grid2d, block2d>>>(bDim, xDim, ti, theta, t, x, a0, ax, axx, v, w, bl, b, bu);
    cudaDeviceSynchronize();

    return "";
}


template
kw::Error
adjustEarlyExercise(
    const dim3 block2d,
    const size_t bDim,
    const size_t xDim,
    const float* pay,
    float* v);

template
kw::Error
adjustEarlyExercise(
    const dim3 block2d,
    const size_t bDim,
    const size_t xDim,
    const double* pay,
    double* v);


template
kw::Error
kw::fillB(
    const dim3 block2d,
    const size_t bDim,
    const size_t xDim,
    const size_t ti,
    const float theta,
    const float* t,
    const float* x,
    const float* a0,
    const float* ax,
    const float* axx,
    const float* v,
    float* w,
    float* bl,
    float* b,
    float* bu);

template
kw::Error
kw::fillB(
    const dim3 block2d,
    const size_t bDim,
    const size_t xDim,
    const size_t ti,
    const double theta,
    const double* t,
    const double* x,
    const double* a0,
    const double* ax,
    const double* axx,
    const double* v,
    double* w,
    double* bl,
    double* b,
    double* bu);

}
